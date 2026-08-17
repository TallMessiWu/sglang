import types
import unittest
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.hardware_backend.npu.moe.lora import (
    build_ascend_moe_lora_indices,
)
from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
    NPUUnquantMoEMethod,
)
from sglang.srt.layers.moe.moe_runner.ascend import (
    AscendRunnerCore,
    AscendRunnerInput,
)
from sglang.srt.lora.layers import _validate_ascend_moe_lora_layer
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.lora_moe_runners import LoRAHooks


class AscendMoeLoraRoutingTest(unittest.TestCase):
    def test_recovers_expert_sorted_lora_rows(self):
        topk_ids = torch.tensor([[2, 0], [1, 2], [0, 1]], dtype=torch.int32)
        # Original flat positions [1, 4, 2, 5, 0, 3] become sorted rows.
        expanded_row_idx = torch.tensor([4, 0, 2, 5, 1, 3], dtype=torch.int32)
        token_lora_mapping = torch.tensor([0, 1, 2], dtype=torch.int32)
        adapter_enabled = torch.tensor([0, 1, 1], dtype=torch.int32)

        combined, components = build_ascend_moe_lora_indices(
            expanded_row_idx,
            topk_ids,
            token_lora_mapping,
            adapter_enabled,
            num_experts=3,
        )

        torch.testing.assert_close(
            combined,
            torch.tensor([-1, 6, 4, 7, -1, 5], dtype=torch.int64),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            components,
            torch.tensor(
                [-1, -1, 12, 13, 8, 9, 14, 15, -1, -1, 10, 11],
                dtype=torch.int64,
            ),
            rtol=0,
            atol=0,
        )

    def test_negative_and_disabled_adapters_use_sentinel(self):
        combined, components = build_ascend_moe_lora_indices(
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([[0, 1]], dtype=torch.int32),
            torch.tensor([-1], dtype=torch.int32),
            torch.tensor([0, 1], dtype=torch.int32),
            num_experts=2,
        )
        self.assertTrue(torch.equal(combined, torch.full((2,), -1)))
        self.assertTrue(torch.equal(components, torch.full((4,), -1)))


class AscendMoeLoraScopeTest(unittest.TestCase):
    def _make_layer(self):
        return types.SimpleNamespace(
            moe_ep_size=1,
            is_shared_fused_moe=False,
            num_fused_shared_experts=0,
            params_dtype=torch.bfloat16,
            w13_kernel=NPUUnquantMoEMethod(),
            w2_kernel=NPUUnquantMoEMethod(),
            moe_runner_config=types.SimpleNamespace(
                is_gated=True,
                activation="silu",
                gemm1_alpha=None,
                gemm1_clamp_limit=None,
            ),
        )

    def _validate(self, layer, backend="none"):
        a2a = types.SimpleNamespace(
            is_none=lambda: backend == "none",
            value=backend,
        )
        with mock.patch(
            "sglang.srt.layers.moe.utils.get_moe_a2a_backend",
            return_value=a2a,
        ):
            _validate_ascend_moe_lora_layer(layer)

    def test_accepts_bf16_unquantized_tp(self):
        self._validate(self._make_layer())

    def test_rejects_ep(self):
        layer = self._make_layer()
        layer.moe_ep_size = 2
        with self.assertRaisesRegex(NotImplementedError, "TP-only"):
            self._validate(layer)

    def test_rejects_alltoall(self):
        with self.assertRaisesRegex(NotImplementedError, "moe-a2a-backend none"):
            self._validate(self._make_layer(), backend="deepep")

    def test_rejects_quantized_experts(self):
        layer = self._make_layer()
        layer.w13_kernel = object()
        with self.assertRaisesRegex(NotImplementedError, "unquantized"):
            self._validate(layer)

    def test_rejects_shared_experts(self):
        layer = self._make_layer()
        layer.num_fused_shared_experts = 1
        with self.assertRaisesRegex(NotImplementedError, "shared experts"):
            self._validate(layer)

        layer = self._make_layer()
        layer.is_shared_fused_moe = True
        with self.assertRaisesRegex(NotImplementedError, "shared experts"):
            self._validate(layer)


class AscendMoeLoraHookOrderTest(unittest.TestCase):
    def test_hooks_run_around_activation_and_down_projection(self):
        class GateKernel:
            def apply(self, _quant_info, hidden_states, *_args, **_kwargs):
                return torch.zeros(
                    hidden_states.shape[0], 6, dtype=hidden_states.dtype
                )

        class Activation:
            def _apply_activation(self, hidden_states):
                torch.testing.assert_close(
                    hidden_states, torch.ones_like(hidden_states)
                )
                return hidden_states[:, :3], None

        class DownKernel:
            def apply(self, _quant_info, hidden_states, *_args, **_kwargs):
                torch.testing.assert_close(
                    hidden_states, torch.ones_like(hidden_states)
                )
                return torch.zeros(
                    hidden_states.shape[0], 4, dtype=hidden_states.dtype
                )

        core = object.__new__(AscendRunnerCore)
        core.config = types.SimpleNamespace(
            layer=types.SimpleNamespace(
                w13_kernel=GateKernel(),
                w2_kernel=DownKernel(),
            )
        )
        core.activation = Activation()
        runner_input = AscendRunnerInput(
            hidden_states=torch.zeros(2, 4, dtype=torch.bfloat16),
            hidden_states_scale=None,
            expert_tokens=torch.tensor([2], dtype=torch.int64),
            group_list_type=1,
            topk_ids=torch.zeros(2, 1, dtype=torch.int32),
            expanded_row_idx=torch.arange(2, dtype=torch.int32),
        )
        hooks = LoRAHooks(
            after_gate_up=lambda _x, output, *_args: output.add_(1),
            after_down=lambda _x, output, *_args: output.add_(2),
        )

        output = core.run(runner_input, quant_info=None, running_state={}, hooks=hooks)
        torch.testing.assert_close(
            output.hidden_states,
            torch.full((2, 4), 2, dtype=torch.bfloat16),
        )


class AscendMoeLoraModelScopeTest(unittest.TestCase):
    def _make_manager(self, config):
        class FakeFusedMoE:
            pass

        manager = types.SimpleNamespace(
            lora_backend=types.SimpleNamespace(name="ascend"),
            target_modules={"gate_up_proj", "down_proj"},
            base_model=types.SimpleNamespace(modules=lambda: [FakeFusedMoE()]),
            lora_use_virtual_experts=False,
            experts_shared_outer_loras=False,
            base_hf_config=config,
        )
        return manager, FakeFusedMoE

    def test_rejects_model_with_separate_shared_experts(self):
        manager, fake_moe = self._make_manager(
            types.SimpleNamespace(
                shared_expert_intermediate_size=1024,
                n_shared_experts=0,
            )
        )
        with mock.patch("sglang.srt.lora.lora_manager.FusedMoE", fake_moe):
            with self.assertRaisesRegex(NotImplementedError, "shared experts"):
                LoRAManager._validate_ascend_moe_lora_scope(manager)

    def test_rejects_virtual_experts(self):
        manager, fake_moe = self._make_manager(types.SimpleNamespace())
        manager.lora_use_virtual_experts = True
        with mock.patch("sglang.srt.lora.lora_manager.FusedMoE", fake_moe):
            with self.assertRaisesRegex(NotImplementedError, "virtual-experts"):
                LoRAManager._validate_ascend_moe_lora_scope(manager)


if __name__ == "__main__":
    unittest.main()
