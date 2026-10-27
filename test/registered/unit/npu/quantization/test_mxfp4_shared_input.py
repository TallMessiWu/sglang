"""CPU call/shape contracts for shared online W4A8 GDN input quantization."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.layers.quantization  # noqa: F401
from sglang.srt.hardware_backend.npu.quantization import linear_method_npu
from sglang.srt.layers import linear
from sglang.srt.layers.linear import MergedColumnParallelLinear
from sglang.srt.layers.quantization import unquant
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.models import qwen3_5
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def make_projection(width, dtype, quantized=True):
    layer = MergedColumnParallelLinear.__new__(MergedColumnParallelLinear)
    torch.nn.Module.__init__(layer)
    layer.bias = None
    layer.skip_bias_add = False
    layer.gather_output = False
    layer.parallel_group = "tp"
    layer.tp_group = SimpleNamespace(rank_in_group=0, world_size=2, rank=0, size=2)
    if quantized:
        layer.quant_method = linear_method_npu.NPUMXFP4W4A8LinearMethod()
        layer.weight = torch.nn.Parameter(
            torch.arange(32 * width).reshape(32, width).remainder(4).to(torch.uint8),
            requires_grad=False,
        )
        layer.weight_scale = torch.nn.Parameter(
            torch.ones(1, width, 2, dtype=torch.uint8), requires_grad=False
        )
    else:
        layer.quant_method = UnquantizedLinearMethod()
        layer.weight = torch.nn.Parameter(
            torch.ones(width, 64, dtype=dtype), requires_grad=False
        )
    return layer


def make_model(dtype, quantized=(True, True)):
    model = qwen3_5.Qwen3_5GatedDeltaNet.__new__(qwen3_5.Qwen3_5GatedDeltaNet)
    torch.nn.Module.__init__(model)
    model.in_proj_qkvz = make_projection(128, dtype, quantized[0])
    model.in_proj_ba = make_projection(8, dtype, quantized[1])
    model._fused_in_proj_weight = None
    model._fused_input_proj_cpu_enabled = SimpleNamespace(value=False)
    model.alt_stream = None
    return model


@contextmanager
def isolated_ops(*, npu=True, lora=False, lora_paths=None, sp=False):
    def quantize(x, **kwargs):
        return x.to(torch.float8_e4m3fn), torch.ones(
            x.shape[0], 1, 2, dtype=torch.uint8
        )

    def matmul(x, weight, scale, **kwargs):
        # Deterministic stand-in to compare the two call paths, not an FP4 decoder.
        expanded = weight.float().repeat_interleave(2, dim=0)
        output = x.float() @ expanded
        if kwargs["bias"] is not None:
            output += kwargs["bias"].float()
        return output.to(kwargs["output_dtype"])

    def sp_matmul(layer, x, bias):
        # A prepared pair must never reach the SP all-gather boundary.
        assert isinstance(x, torch.Tensor)
        return layer.quant_method.apply(layer, x, bias)

    with (
        patch.object(qwen3_5, "_is_npu", npu),
        patch.object(qwen3_5, "_is_cpu", False),
        patch.object(qwen3_5, "_use_aiter", False),
        patch.object(
            qwen3_5,
            "get_lora",
            return_value=SimpleNamespace(enable_lora=lora, lora_paths=lora_paths),
        ),
        patch.object(
            qwen3_5, "get_forward", return_value=SimpleNamespace(sp_active=sp)
        ),
        patch.object(qwen3_5, "check_cuda_graph_backend", return_value=False),
        patch.object(linear, "get_forward", return_value=SimpleNamespace(sp_active=sp)),
        patch.object(
            linear, "layernorm_sp", SimpleNamespace(column_parallel_g_matmul=sp_matmul)
        ),
        patch.object(unquant, "use_intel_amx_backend", return_value=False),
        patch.object(
            unquant,
            "get_bf16_gemm_backend",
            return_value=SimpleNamespace(
                is_cutedsl=lambda: False, is_gemv=lambda: False
            ),
        ),
        patch.object(unquant, "_use_aiter", False),
        patch.object(
            linear_method_npu, "_get_float4_e2m1fn_x2_dtype", return_value=296
        ),
        patch.object(linear_method_npu, "_get_float8_e8m0fnu_dtype", return_value=293),
        patch.object(
            torch.ops.npu, "npu_dynamic_mx_quant", side_effect=quantize, create=True
        ) as quant,
        patch.object(
            torch.ops.npu, "npu_quant_matmul", side_effect=matmul, create=True
        ) as mm,
    ):
        yield quant, mm


class TestMXFP4SharedInput(unittest.TestCase):
    def test_gdn_quantizes_once_and_matches_separate_projections(self):
        for dtype in (torch.bfloat16, torch.float16):
            for tokens in (0, 1, 32, 128):
                with (
                    self.subTest(dtype=dtype, tokens=tokens),
                    isolated_ops() as (quant, mm),
                ):
                    model = make_model(dtype)
                    # Non-contiguous input also exercises flattening without changing shape.
                    x = (
                        torch.arange(tokens * 128, dtype=torch.float32)
                        .reshape(tokens, 128)
                        .remainder(8)
                        .to(dtype)[:, ::2]
                    )
                    expected = (model.in_proj_qkvz(x)[0], model.in_proj_ba(x)[0])
                    self.assertEqual(quant.call_count, 2)
                    quant.reset_mock()
                    mm.reset_mock()
                    actual = model._forward_input_proj(x)
                    self.assertEqual(quant.call_count, 1)
                    self.assertEqual(mm.call_count, 2)
                    self.assertIs(
                        mm.call_args_list[0].args[0], mm.call_args_list[1].args[0]
                    )
                    self.assertIs(
                        mm.call_args_list[0].kwargs["pertoken_scale"],
                        mm.call_args_list[1].kwargs["pertoken_scale"],
                    )
                    for result, reference in zip(actual, expected):
                        self.assertEqual(result.dtype, dtype)
                        torch.testing.assert_close(result, reference, rtol=0, atol=0)

    def test_prepared_linear_preserves_shapes_dtypes_and_bias(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for shape in ((3, 64), (2, 3, 64)):
                with (
                    self.subTest(dtype=dtype, shape=shape),
                    isolated_ops() as (quant, mm),
                ):
                    layer = make_projection(8, dtype)
                    x = torch.ones(shape, dtype=dtype)
                    bias = torch.arange(8, dtype=torch.float32)
                    reference = layer.quant_method.apply(layer, x, bias)
                    quant.reset_mock()
                    prepared = layer.quant_method.prepare_input(x)
                    self.assertEqual(quant.call_count, 1)
                    quant.reset_mock()
                    result = layer.quant_method.apply(layer, prepared, bias)
                    quant.assert_not_called()
                    self.assertEqual(result.shape, shape[:-1] + (8,))
                    self.assertEqual(
                        result.dtype,
                        torch.bfloat16 if dtype == torch.float32 else dtype,
                    )
                    self.assertEqual(mm.call_args.kwargs["bias"].shape, (1, 8))
                    self.assertEqual(mm.call_args.kwargs["bias"].dtype, torch.bfloat16)
                    self.assertEqual(mm.call_args.kwargs["group_sizes"], [0, 0, 32])
                    self.assertEqual(mm.call_args.kwargs["x2_dtype"], 296)
                    self.assertIs(mm.call_args.args[0], prepared.quantized)
                    torch.testing.assert_close(result, reference, rtol=0, atol=0)

    def test_shared_inputs_are_not_cached_between_forwards(self):
        model = make_model(torch.bfloat16)
        x = torch.ones(4, 64, dtype=torch.bfloat16)
        with isolated_ops() as (quant, mm):
            before = model._forward_input_proj(x)
            x.mul_(2)
            after = model._forward_input_proj(x)
        self.assertEqual(quant.call_count, 2)
        self.assertIsNot(mm.call_args_list[0].args[0], mm.call_args_list[2].args[0])
        for first, second in zip(before, after):
            torch.testing.assert_close(second, first * 2, rtol=0, atol=0)

    def test_mixed_and_unquantized_projections_keep_independent_paths(self):
        for modes in ((True, False), (False, True), (False, False)):
            with self.subTest(modes=modes), isolated_ops() as (quant, _):
                model = make_model(torch.bfloat16, modes)
                output = model._forward_input_proj(
                    torch.ones(4, 64, dtype=torch.bfloat16)
                )
                self.assertEqual(quant.call_count, sum(modes))
                self.assertEqual(output[0].shape, (4, 128))
                self.assertEqual(output[1].shape, (4, 8))

    def test_lora_sp_and_other_platforms_keep_tensor_inputs(self):
        for options in (
            {"lora": True},
            {"lora_paths": ["adapter"]},
            {"sp": True},
            {"npu": False},
        ):
            with self.subTest(options=options), isolated_ops(**options) as (quant, _):
                model = make_model(torch.bfloat16)
                inputs = []
                model.in_proj_qkvz.register_forward_pre_hook(
                    lambda _, args: inputs.append(args[0])
                )
                model.in_proj_ba.register_forward_pre_hook(
                    lambda _, args: inputs.append(args[0])
                )
                model._forward_input_proj(torch.ones(4, 64, dtype=torch.bfloat16))
                self.assertEqual(quant.call_count, 2)
                self.assertTrue(all(isinstance(x, torch.Tensor) for x in inputs))

    def test_offline_and_custom_methods_do_not_share_online_inputs(self):
        class CustomMethod(linear_method_npu.NPUMXFP4W4A8LinearMethod):
            pass

        for method_type in (
            linear_method_npu.NPUMXFP4W4A8OfflineLinearMethod,
            CustomMethod,
        ):
            with self.subTest(method=method_type), isolated_ops() as (quant, _):
                model = make_model(torch.bfloat16)
                model.in_proj_qkvz.quant_method = method_type()
                model.in_proj_ba.quant_method = method_type()
                model._forward_input_proj(torch.ones(4, 64, dtype=torch.bfloat16))
                self.assertEqual(quant.call_count, 2)


if __name__ == "__main__":
    unittest.main()
