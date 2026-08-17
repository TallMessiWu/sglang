"""Ascend TP-only MoE LoRA hooks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.lora.lora_moe_runners import LoRAHooks
from sglang.srt.model_executor.runner import get_is_capture_mode

if TYPE_CHECKING:
    from sglang.srt.layers.moe.moe_runner.ascend import AscendRunnerInput
    from sglang.srt.lora.lora_moe_runners import LoRAInfo


def build_ascend_moe_lora_indices(
    expanded_row_idx: torch.Tensor,
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    adapter_enabled: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map expert-sorted rows to LoRA/expert and gate/up component slots."""
    expanded = expanded_row_idx.abs()
    inverse_permutation = torch.argsort(expanded).to(torch.long)
    expert_per_row = topk_ids.reshape(-1).to(torch.long)[inverse_permutation]
    token_per_row = inverse_permutation // topk_ids.shape[1]

    lora_per_row = token_lora_mapping.to(torch.long)[token_per_row]
    safe_lora = lora_per_row.clamp(min=0)
    enabled = (lora_per_row >= 0) & adapter_enabled.to(torch.bool)[safe_lora]
    combined = torch.where(
        enabled,
        safe_lora * num_experts + expert_per_row,
        torch.full_like(safe_lora, -1),
    ).contiguous()

    component_offsets = torch.arange(2, dtype=torch.long, device=combined.device)
    component_indices = torch.where(
        combined[:, None] >= 0,
        combined[:, None] * 2 + component_offsets,
        torch.full_like(combined[:, None], -1),
    )
    return combined, component_indices.reshape(-1).contiguous()


def _validate_lora_info(lora_info: LoRAInfo) -> None:
    if lora_info.experts_shared_outer_loras:
        raise NotImplementedError(
            "Ascend MoE LoRA MVP does not support shared-outer expert weights."
        )
    if lora_info.lora_use_virtual_experts:
        raise NotImplementedError(
            "Ascend MoE LoRA MVP does not support virtual experts."
        )
    if lora_info.fully_sharded:
        raise NotImplementedError(
            "Ascend MoE LoRA MVP does not support fully-sharded LoRA weights."
        )

    rank = lora_info.max_lora_rank
    gate_a = lora_info.gate_up_lora_a_weights
    gate_b = lora_info.gate_up_lora_b_weights
    down_a = lora_info.down_lora_a_weights
    down_b = lora_info.down_lora_b_weights
    weights = (gate_a, gate_b, down_a, down_b)
    if any(weight is None or weight.ndim != 4 for weight in weights):
        raise ValueError(
            "Ascend MoE LoRA requires four per-expert 4D weight buffers."
        )
    if rank not in (8, 16, 32, 64):
        raise NotImplementedError(
            "Ascend MoE LoRA BGMV supports max LoRA rank 8, 16, 32, or 64."
        )
    if any(weight.dtype != torch.bfloat16 for weight in weights):
        raise NotImplementedError(
            "Ascend MoE LoRA MVP supports BF16 adapter weights only."
        )
    num_loras = gate_a.shape[0]
    if any(weight.shape[0] != num_loras for weight in weights):
        raise ValueError("Ascend MoE LoRA weight buffers disagree on adapter slots.")
    if lora_info.adapter_enabled.shape[0] != num_loras:
        raise ValueError(
            "Ascend MoE LoRA adapter mask does not match the weight buffers."
        )
    if any(weight.shape[1] != lora_info.num_experts for weight in weights):
        raise ValueError(
            "Ascend MoE LoRA requires one LoRA weight set per local expert."
        )
    if gate_a.shape[2] != 2 * rank or gate_b.shape[-1] != rank:
        raise ValueError(
            "Ascend MoE LoRA requires gated W13 weights with A=2*rank and "
            "B=rank."
        )
    if gate_b.shape[2] % 2 != 0:
        raise ValueError("Ascend MoE LoRA W13 output dimension must be even.")
    if down_a.shape[2] != rank or down_b.shape[-1] != rank:
        raise ValueError(
            "Ascend MoE LoRA requires W2 weights with matching max LoRA rank."
        )
    if gate_b.shape[2] // 2 != down_a.shape[3]:
        raise ValueError("Ascend MoE LoRA W13 and W2 intermediate sizes disagree.")
    if gate_a.shape[3] != down_b.shape[2]:
        raise ValueError("Ascend MoE LoRA W13 and W2 hidden sizes disagree.")


def _get_shrink_buffer(
    hidden_states: torch.Tensor,
    lora_info: LoRAInfo,
    width: int,
) -> torch.Tensor:
    rows = hidden_states.shape[0]
    if get_is_capture_mode() and lora_info.cg_buffers is not None:
        buffer = lora_info.cg_buffers["ascend_shrink_buffer"]
        if rows > buffer.shape[0] or width > buffer.shape[1]:
            raise ValueError(
                "Ascend MoE LoRA NPU Graph scratch buffer is smaller than the "
                "captured batch."
            )
        return buffer[:rows, :width].zero_()
    return torch.zeros(
        (rows, width), dtype=torch.float32, device=hidden_states.device
    )


def _add_gate_up_delta(
    hidden_states: torch.Tensor,
    gate_up_output: torch.Tensor,
    component_indices: torch.Tensor,
    lora_info: LoRAInfo,
) -> None:
    rank = lora_info.max_lora_rank
    rows = hidden_states.shape[0]
    intermediate_size = gate_up_output.shape[-1] // 2
    expanded_hidden_states = (
        hidden_states[:, None, :]
        .expand(-1, 2, -1)
        .reshape(rows * 2, -1)
        .contiguous()
    )
    shrink = _get_shrink_buffer(expanded_hidden_states, lora_info, rank)

    gate_a = lora_info.gate_up_lora_a_weights.reshape(
        -1, 2, rank, hidden_states.shape[-1]
    ).reshape(-1, rank, hidden_states.shape[-1])
    gate_b = lora_info.gate_up_lora_b_weights.reshape(
        -1, 2, intermediate_size, rank
    ).reshape(-1, intermediate_size, rank)

    torch.ops.npu.bgmv_shrink(
        expanded_hidden_states,
        gate_a,
        component_indices,
        shrink,
        1.0,
    )
    torch.ops.npu.bgmv_expand(
        shrink,
        gate_b,
        component_indices,
        gate_up_output.reshape(rows * 2, intermediate_size),
        0,
        intermediate_size,
    )


def _add_down_delta(
    intermediate_states: torch.Tensor,
    down_output: torch.Tensor,
    combined_indices: torch.Tensor,
    lora_info: LoRAInfo,
) -> None:
    rank = lora_info.max_lora_rank
    rows = intermediate_states.shape[0]
    shrink = _get_shrink_buffer(intermediate_states, lora_info, rank)
    down_a = lora_info.down_lora_a_weights.reshape(
        -1, rank, intermediate_states.shape[-1]
    )
    down_b = lora_info.down_lora_b_weights.reshape(
        -1, down_output.shape[-1], rank
    )

    torch.ops.npu.bgmv_shrink(
        intermediate_states.reshape(rows, -1),
        down_a,
        combined_indices,
        shrink,
        1.0,
    )
    torch.ops.npu.bgmv_expand(
        shrink,
        down_b,
        combined_indices,
        down_output.reshape(rows, -1),
        0,
        down_output.shape[-1],
    )


def build_ascend_moe_lora_hooks(
    runner_input: AscendRunnerInput,
    lora_info: LoRAInfo,
) -> LoRAHooks:
    """Build fixed-shape BGMV hooks for Ascend TP dispatch."""
    if lora_info is None or lora_info.max_lora_rank == 0:
        return LoRAHooks()
    if not get_is_capture_mode() and not lora_info.has_active_lora:
        return LoRAHooks()
    if runner_input.topk_ids is None or runner_input.expanded_row_idx is None:
        raise NotImplementedError(
            "Ascend MoE LoRA MVP requires Ascend TP routing metadata; "
            "EP/AlltoAll dispatch is not supported."
        )

    _validate_lora_info(lora_info)
    if runner_input.hidden_states.shape[0] != runner_input.topk_ids.numel():
        raise ValueError(
            "Ascend MoE LoRA routing metadata does not match the dispatched rows."
        )
    if lora_info.token_lora_mapping.shape[0] < runner_input.topk_ids.shape[0]:
        raise ValueError(
            "Ascend MoE LoRA token mapping is shorter than the routed token batch."
        )

    combined_indices, component_indices = build_ascend_moe_lora_indices(
        runner_input.expanded_row_idx,
        runner_input.topk_ids,
        lora_info.token_lora_mapping,
        lora_info.adapter_enabled,
        lora_info.num_experts,
    )

    def after_gate_up(hidden_states, gate_up_output, _topk_weights, _topk_ids):
        _add_gate_up_delta(
            hidden_states,
            gate_up_output,
            component_indices,
            lora_info,
        )

    def after_down(intermediate_states, down_output, _topk_weights, _topk_ids):
        _add_down_delta(
            intermediate_states,
            down_output,
            combined_indices,
            lora_info,
        )

    return LoRAHooks(after_gate_up=after_gate_up, after_down=after_down)
