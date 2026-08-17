import types
import unittest

import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=15, suite="stage-a-unit-test-npu")

import sgl_kernel_npu  # noqa: F401
import torch_npu  # noqa: F401

from sglang.srt.hardware_backend.npu.moe.lora import (
    build_ascend_moe_lora_hooks,
)


class AscendMoeLoraBgmvTest(unittest.TestCase):
    def _make_case(self, rank):
        torch.manual_seed(7)
        device = torch.device("npu")
        num_loras, num_experts = 3, 3
        hidden_size, intermediate_size = 256, 128
        topk_ids = torch.tensor(
            [[2, 0], [1, 2], [0, 1]], dtype=torch.int32, device=device
        )
        expanded_row_idx = torch.tensor(
            [4, 0, 2, 5, 1, 3], dtype=torch.int32, device=device
        )
        token_lora_mapping = torch.tensor(
            [0, 1, 2], dtype=torch.int32, device=device
        )
        adapter_enabled = torch.tensor(
            [0, 1, 1], dtype=torch.int32, device=device
        )
        rows = topk_ids.numel()
        hidden_states = torch.randn(
            rows, hidden_size, dtype=torch.bfloat16, device=device
        )
        info = types.SimpleNamespace(
            max_lora_rank=rank,
            num_experts=num_experts,
            has_active_lora=True,
            experts_shared_outer_loras=False,
            lora_use_virtual_experts=False,
            fully_sharded=False,
            cg_buffers=None,
            token_lora_mapping=token_lora_mapping,
            adapter_enabled=adapter_enabled,
            gate_up_lora_a_weights=torch.randn(
                num_loras,
                num_experts,
                2 * rank,
                hidden_size,
                dtype=torch.bfloat16,
                device=device,
            ).mul_(0.05),
            gate_up_lora_b_weights=torch.randn(
                num_loras,
                num_experts,
                2 * intermediate_size,
                rank,
                dtype=torch.bfloat16,
                device=device,
            ).mul_(0.05),
            down_lora_a_weights=torch.randn(
                num_loras,
                num_experts,
                rank,
                intermediate_size,
                dtype=torch.bfloat16,
                device=device,
            ).mul_(0.05),
            down_lora_b_weights=torch.randn(
                num_loras,
                num_experts,
                hidden_size,
                rank,
                dtype=torch.bfloat16,
                device=device,
            ).mul_(0.05),
        )
        runner_input = types.SimpleNamespace(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            expanded_row_idx=expanded_row_idx,
        )
        return runner_input, info

    @staticmethod
    def _route_slots(info, runner_input):
        inverse = torch.argsort(runner_input.expanded_row_idx.abs()).cpu()
        experts = runner_input.topk_ids.reshape(-1).cpu()[inverse]
        tokens = inverse // runner_input.topk_ids.shape[1]
        loras = info.token_lora_mapping.cpu()[tokens]
        return list(zip(loras.tolist(), experts.tolist()))

    @unittest.skipUnless(torch.npu.is_available(), "Ascend NPU is required")
    def test_gate_up_and_down_match_reference(self):
        for rank in (8, 16, 64):
            runner_input, info = self._make_case(rank)
            hooks = build_ascend_moe_lora_hooks(runner_input, info)
            routes = self._route_slots(info, runner_input)

            gate_output = torch.zeros(
                runner_input.hidden_states.shape[0],
                info.gate_up_lora_b_weights.shape[2],
                dtype=torch.bfloat16,
                device="npu",
            )
            hooks.after_gate_up(
                runner_input.hidden_states,
                gate_output,
                None,
                runner_input.topk_ids,
            )
            gate_reference = torch.zeros(gate_output.shape, dtype=torch.float32)
            intermediate_size = gate_output.shape[1] // 2
            enabled = info.adapter_enabled.cpu()
            for row, (lora_id, expert_id) in enumerate(routes):
                if not enabled[lora_id].item():
                    continue
                x = runner_input.hidden_states[row].float().cpu()
                a = (
                    info.gate_up_lora_a_weights[lora_id, expert_id].float().cpu()
                )
                b = (
                    info.gate_up_lora_b_weights[lora_id, expert_id].float().cpu()
                )
                gate_reference[row, :intermediate_size] = (
                    x @ a[:rank].T @ b[:intermediate_size].T
                )
                gate_reference[row, intermediate_size:] = (
                    x @ a[rank:].T @ b[intermediate_size:].T
                )
            torch.testing.assert_close(
                gate_output.float().cpu(),
                gate_reference,
                atol=2e-2,
                rtol=2e-2,
            )

            intermediate = F.silu(gate_output[:, :intermediate_size].float())
            intermediate *= gate_output[:, intermediate_size:].float()
            intermediate = intermediate.to(torch.bfloat16)
            down_output = torch.zeros(
                intermediate.shape[0],
                info.down_lora_b_weights.shape[2],
                dtype=torch.bfloat16,
                device="npu",
            )
            hooks.after_down(intermediate, down_output, None, runner_input.topk_ids)
            down_reference = torch.zeros(down_output.shape, dtype=torch.float32)
            for row, (lora_id, expert_id) in enumerate(routes):
                if not enabled[lora_id].item():
                    continue
                x = intermediate[row].float().cpu()
                a = info.down_lora_a_weights[lora_id, expert_id].float().cpu()
                b = info.down_lora_b_weights[lora_id, expert_id].float().cpu()
                down_reference[row] = x @ a.T @ b.T
            torch.testing.assert_close(
                down_output.float().cpu(),
                down_reference,
                atol=2e-2,
                rtol=2e-2,
            )

    @unittest.skipUnless(torch.npu.is_available(), "Ascend NPU is required")
    def test_small_moe_layer_matches_explicit_weight_delta(self):
        runner_input, info = self._make_case(rank=8)
        hooks = build_ascend_moe_lora_hooks(runner_input, info)
        routes = self._route_slots(info, runner_input)
        expert_ids = torch.tensor(
            [expert_id for _, expert_id in routes],
            dtype=torch.long,
            device="npu",
        )
        hidden_size = runner_input.hidden_states.shape[1]
        intermediate_size = info.down_lora_a_weights.shape[-1]
        base_w13 = torch.randn(
            info.num_experts,
            2 * intermediate_size,
            hidden_size,
            dtype=torch.bfloat16,
            device="npu",
        ).mul_(0.05)
        base_w2 = torch.randn(
            info.num_experts,
            hidden_size,
            intermediate_size,
            dtype=torch.bfloat16,
            device="npu",
        ).mul_(0.05)

        gate_up = torch.bmm(
            base_w13[expert_ids], runner_input.hidden_states.unsqueeze(-1)
        ).squeeze(-1)
        hooks.after_gate_up(
            runner_input.hidden_states,
            gate_up,
            None,
            runner_input.topk_ids,
        )
        intermediate = F.silu(gate_up[:, :intermediate_size].float())
        intermediate *= gate_up[:, intermediate_size:].float()
        intermediate = intermediate.to(torch.bfloat16)
        output = torch.bmm(
            base_w2[expert_ids], intermediate.unsqueeze(-1)
        ).squeeze(-1)
        hooks.after_down(intermediate, output, None, runner_input.topk_ids)

        reference = torch.zeros(output.shape, dtype=torch.float32)
        enabled = info.adapter_enabled.cpu()
        for row, (lora_id, expert_id) in enumerate(routes):
            x = runner_input.hidden_states[row].float().cpu()
            gate = x @ base_w13[expert_id].float().cpu().T
            if enabled[lora_id].item():
                gate_a = (
                    info.gate_up_lora_a_weights[lora_id, expert_id].float().cpu()
                )
                gate_b = (
                    info.gate_up_lora_b_weights[lora_id, expert_id].float().cpu()
                )
                gate[:intermediate_size] += (
                    x @ gate_a[:8].T @ gate_b[:intermediate_size].T
                )
                gate[intermediate_size:] += (
                    x @ gate_a[8:].T @ gate_b[intermediate_size:].T
                )
            activated = F.silu(gate[:intermediate_size])
            activated *= gate[intermediate_size:]
            down = activated @ base_w2[expert_id].float().cpu().T
            if enabled[lora_id].item():
                down_a = (
                    info.down_lora_a_weights[lora_id, expert_id].float().cpu()
                )
                down_b = (
                    info.down_lora_b_weights[lora_id, expert_id].float().cpu()
                )
                down += activated @ down_a.T @ down_b.T
            reference[row] = down

        torch.testing.assert_close(
            output.float().cpu(),
            reference,
            atol=2e-2,
            rtol=2e-2,
        )

    @unittest.skipUnless(torch.npu.is_available(), "Ascend NPU is required")
    def test_all_disabled_adapters_leave_outputs_bitwise_identical(self):
        runner_input, info = self._make_case(rank=8)
        info.adapter_enabled.zero_()
        hooks = build_ascend_moe_lora_hooks(runner_input, info)

        gate_output = torch.randn(
            runner_input.hidden_states.shape[0],
            info.gate_up_lora_b_weights.shape[2],
            dtype=torch.bfloat16,
            device="npu",
        )
        expected = gate_output.clone()
        hooks.after_gate_up(
            runner_input.hidden_states,
            gate_output,
            None,
            runner_input.topk_ids,
        )
        self.assertTrue(torch.equal(gate_output, expected))

        intermediate = torch.randn(
            runner_input.hidden_states.shape[0],
            info.down_lora_a_weights.shape[-1],
            dtype=torch.bfloat16,
            device="npu",
        )
        down_output = torch.randn(
            runner_input.hidden_states.shape[0],
            info.down_lora_b_weights.shape[2],
            dtype=torch.bfloat16,
            device="npu",
        )
        expected = down_output.clone()
        hooks.after_down(intermediate, down_output, None, runner_input.topk_ids)
        self.assertTrue(torch.equal(down_output, expected))


if __name__ == "__main__":
    unittest.main()
