"""CPU contracts for MXFP4 weights with routing-fused MXFP8 activations."""

import unittest
from itertools import product
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.layers.quantization  # noqa: F401
from sglang.srt.hardware_backend.npu.quantization import moe_methods
from sglang.srt.layers.moe import utils as moe_utils
from sglang.srt.layers.moe.moe_runner.ascend import (
    AscendRunnerCore,
    AscendRunnerInput,
    pre_permute_ascend_tp_to_ascend,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.ascend_tp import AscendTPDispatcher
from sglang.srt.layers.moe.utils import DeepEPMode, DispatcherOutputDtype
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMXFP4Routing(unittest.TestCase):
    def test_online_and_offline_routing_reuse_scales_and_preserve_dtype(self):
        for online, fuse_routing in product((False, True), repeat=2):
            for dtype in (torch.bfloat16, torch.float16):
                with self.subTest(online=online, dtype=dtype, fused=fuse_routing):
                    layer = torch.nn.Module()
                    layer.w13_kernel = moe_methods.NPUW4A8MXFP4MoEMethod()
                    layer.w2_kernel = moe_methods.NPUW4A8MXFP4MoEMethod()
                    config = MoeRunnerConfig(
                        layer=layer, num_experts=2, top_k=2, params_dtype=dtype
                    )
                    layer.dispatcher = AscendTPDispatcher(config)
                    packed = torch.zeros(2, 128, 32, dtype=torch.uint8)
                    scale = torch.zeros(2, 128, 2, dtype=torch.uint8)
                    layer.w13_weight = torch.nn.Parameter(
                        torch.zeros(2, 128, 64, dtype=dtype) if online else packed,
                        requires_grad=False,
                    )
                    if not online:
                        layer.w13_weight_scale = torch.nn.Parameter(
                            scale, requires_grad=False
                        )
                    layer.w2_weight = torch.nn.Parameter(
                        torch.zeros(2, 32, 64, dtype=torch.uint8), requires_grad=False
                    )
                    layer.w2_weight_scale = torch.nn.Parameter(
                        torch.zeros(2, 1, 64, 2, dtype=torch.uint8), requires_grad=False
                    )
                    with (
                        patch.object(
                            moe_methods.envs.SGLANG_NPU_W4A8_MOE_FUSE_ROUTING_QUANT,
                            "get",
                            return_value=fuse_routing,
                        ),
                        patch.object(
                            moe_methods, "_get_float4_e2m1fn_x2_dtype", return_value=296
                        ),
                        patch.object(
                            moe_methods, "_require_e8m0_dtype", return_value=293
                        ),
                        patch.object(
                            moe_methods, "npu_format_cast", side_effect=lambda x, **_: x
                        ),
                        patch.object(
                            layer.w13_kernel,
                            "_quantize_weight_online",
                            return_value=(packed, scale.reshape(2, 128, 1, 2)),
                        ),
                    ):
                        layer.w13_kernel.process_weights_after_loading(layer, "w13")

                        quantized_x = torch.zeros(4, 64, dtype=torch.float8_e4m3fn)
                        quantized_scale = torch.ones(4, 1, 2, dtype=torch.uint8)
                        routed_x = (
                            quantized_x
                            if fuse_routing
                            else torch.zeros(4, 64, dtype=dtype)
                        )
                        routed_scale = (
                            quantized_scale.flatten(-2) if fuse_routing else None
                        )
                        quant_config = layer.dispatcher.quant_config
                        self.assertEqual(
                            quant_config["normal_dispatcher_output_dtype"], "bf16"
                        )
                        self.assertEqual(
                            quant_config["low_latency_dispatcher_output_dtype"], "mxfp8"
                        )
                        counts = torch.tensor([2, 2], dtype=torch.int64)
                        gmm1_out = torch.zeros(4, 64, dtype=torch.float8_e4m3fn)
                        gmm1_scale = torch.full((4, 1, 2), 2, dtype=torch.uint8)
                        with (
                            patch.object(
                                torch.ops.npu,
                                "npu_moe_init_routing_v2",
                                return_value=(
                                    routed_x,
                                    torch.arange(4),
                                    counts,
                                    routed_scale,
                                ),
                                create=True,
                            ) as routing,
                            patch.object(
                                torch.ops.npu,
                                "npu_dynamic_mx_quant",
                                return_value=(quantized_x, quantized_scale),
                                create=True,
                            ) as quantize,
                            patch.object(
                                torch.ops.npu,
                                "npu_grouped_matmul_swiglu_quant_v2",
                                return_value=(gmm1_out, gmm1_scale),
                                create=True,
                            ) as gmm1,
                            patch.object(
                                torch.ops.npu,
                                "npu_grouped_matmul",
                                side_effect=lambda *args, **kwargs: [
                                    torch.zeros(4, 64, dtype=kwargs["output_dtype"])
                                ],
                                create=True,
                            ) as gmm2,
                        ):
                            dispatch_out = layer.dispatcher.dispatch(
                                torch.zeros(2, 64, dtype=dtype),
                                (
                                    torch.ones(2, 2),
                                    torch.tensor([[0, 1], [1, 0]]),
                                    None,
                                ),
                            )
                            self.assertEqual(
                                routing.call_args.kwargs["quant_mode"],
                                3 if fuse_routing else -1,
                            )
                            runner_input = pre_permute_ascend_tp_to_ascend(
                                dispatch_out, layer, config, {}
                            )
                            runner = AscendRunnerCore(config)
                            self.assertIsNone(runner.activation)
                            result = runner.run(runner_input, layer, {})

                        self.assertEqual(result.hidden_states.dtype, dtype)
                        self.assertEqual(quantize.call_count, 0 if fuse_routing else 1)
                        if not fuse_routing:
                            self.assertIs(quantize.call_args.args[0], routed_x)
                            self.assertEqual(
                                quantize.call_args.kwargs["dst_type"],
                                torch.float8_e4m3fn,
                            )
                        self.assertEqual(gmm1.call_count, 1)
                        self.assertEqual(gmm2.call_count, 1)
                        first, second = gmm1.call_args.kwargs, gmm2.call_args.kwargs
                        self.assertIs(first["x"], quantized_x)
                        self.assertEqual(
                            first["x_scale"].data_ptr(), quantized_scale.data_ptr()
                        )
                        self.assertEqual(first["x_scale"].shape, (4, 1, 2))
                        self.assertTrue(
                            torch.equal(first["group_list"], counts.cumsum(0))
                        )
                        self.assertEqual(first["weight_dtype"], 296)
                        self.assertEqual(
                            second["per_token_scale"][0].data_ptr(),
                            gmm1_scale.data_ptr(),
                        )
                        self.assertIs(second["group_list"], counts)
                        self.assertEqual(second["group_list_type"], 1)

    def test_unquantized_tp_dispatch_keeps_default_and_empty_input(self):
        dispatcher = AscendTPDispatcher(MoeRunnerConfig(num_experts=2, top_k=2))
        x = torch.empty(0, 64, dtype=torch.float16)
        with patch.object(
            torch.ops.npu,
            "npu_moe_init_routing_v2",
            return_value=(x, torch.empty(0), torch.zeros(2, dtype=torch.int64), None),
            create=True,
        ) as routing:
            output = dispatcher.dispatch(
                x, (torch.empty(0, 2), torch.empty(0, 2, dtype=torch.int32), None)
            )
        self.assertEqual(routing.call_args.kwargs["quant_mode"], -1)
        self.assertEqual(output.original_dtype, torch.float16)
        self.assertIs(output.hidden_states, x)
        self.assertIsNone(output.hidden_states_scale)

    def test_runner_without_source_metadata_keeps_existing_dtype_fallback(self):
        layer = SimpleNamespace(
            w13_kernel=moe_methods.NPUW4A8MXFP4MoEMethod(),
            w2_kernel=moe_methods.NPUW4A8MXFP4MoEMethod(),
        )
        runner = AscendRunnerCore(MoeRunnerConfig(layer=layer))
        for payload_dtype, output_dtype in (
            (torch.float16, torch.float16),
            (torch.bfloat16, torch.bfloat16),
            (torch.float8_e4m3fn, torch.bfloat16),
        ):
            with self.subTest(dtype=payload_dtype):
                runner_input = AscendRunnerInput(
                    hidden_states=torch.zeros(2, 64, dtype=payload_dtype),
                    hidden_states_scale=None,
                    expert_tokens=torch.tensor([2], dtype=torch.int64),
                    group_list_type=1,
                )
                with (
                    patch.object(
                        layer.w13_kernel,
                        "apply_fused_gmm1_swiglu",
                        return_value=(
                            torch.zeros(2, 64, dtype=torch.float8_e4m3fn),
                            torch.ones(2, 1, 2, dtype=torch.uint8),
                        ),
                    ),
                    patch.object(
                        layer.w2_kernel, "apply", return_value=torch.empty(2, 64)
                    ) as gmm2,
                ):
                    runner.run(runner_input, layer, {})
                self.assertEqual(gmm2.call_args.kwargs["output_dtype"], output_dtype)

    def test_deepep_mode_overrides_remain_independent_of_tp_default(self):
        # The TP choice must not change either DeepEP mode. Explicit user
        # overrides still win over mode-specific defaults.
        config = {
            "dispatcher_output_dtype": "mxfp8",
            "normal_dispatcher_output_dtype": "bf16",
            "low_latency_dispatcher_output_dtype": "mxfp8",
        }
        with (
            patch.object(moe_utils, "get_server_args", return_value=None),
            patch.object(
                moe_utils.envs.SGLANG_DEEPEP_BF16_DISPATCH, "get", return_value=False
            ),
        ):
            for tp_dtype, (mode, expected) in product(
                ("bf16", "mxfp8"),
                (
                    (DeepEPMode.NORMAL, DispatcherOutputDtype.BF16),
                    (DeepEPMode.LOW_LATENCY, DispatcherOutputDtype.MXFP8),
                ),
            ):
                config["dispatcher_output_dtype"] = tp_dtype
                dispatcher = SimpleNamespace(quant_config=config, dispatch_mode=mode)
                self.assertEqual(
                    moe_utils.get_deepep_output_dtype(dispatcher), expected
                )
                with (
                    patch.object(moe_utils, "get_server_args", return_value=object()),
                    patch.object(
                        moe_utils,
                        "get_exec",
                        return_value=SimpleNamespace(
                            moe=SimpleNamespace(deepep_dispatcher_output_dtype="bf16")
                        ),
                    ),
                ):
                    self.assertEqual(
                        moe_utils.get_deepep_output_dtype(dispatcher),
                        DispatcherOutputDtype.BF16,
                    )


if __name__ == "__main__":
    unittest.main()
