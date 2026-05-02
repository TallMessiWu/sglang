from typing import TYPE_CHECKING, Optional

import torch
import torch_npu
from torch.nn.parameter import Parameter

from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import (
    npu_fused_experts_mxfp8,
)
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
from sglang.srt.utils import set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.layers.quantization.base_config import QuantizationConfig


class NPUMXFP8FusedMoEMethod(FusedMoEMethodBase):
    """Ascend NPU MXFP8 W8A8 online MoE method for Qwen3 MoE and similar models.

    Online mode: loads BF16/FP16 weights, quantises to MXFP8 (float8_e4m3fn) with
    UE8M0 block scales (block_size=32) in process_weights_after_loading.
    FusedMoE (TP-only) path only; EPMoE not yet implemented.
    """

    def __init__(self, quant_config: Optional["QuantizationConfig"] = None):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        # Online quant: register BF16 placeholders; scales created in process_weights_after_loading.
        # Shape convention matches UnquantizedFusedMoEMethod (gated SwiGLU, no triton-kernel transpose).
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: "MoeRunnerConfig"
    ) -> None:
        self.moe_runner_config = moe_runner_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        device = f"npu:{torch.npu.current_device()}"
        num_experts = layer.w13_weight.data.shape[0]

        # Quantise each expert's weights online: BF16 → float8_e4m3fn + uint8 (UE8M0) scales.
        # npu_dynamic_mx_quant expects 2D input [N, K], so we process per expert.
        qw13_list, s13_list = [], []
        for e in range(num_experts):
            w_e = layer.w13_weight.data[e]  # [2I, H]
            if w_e.dtype not in (torch.float16, torch.bfloat16):
                w_e = w_e.to(torch.bfloat16)
            if not w_e.is_npu:
                w_e = w_e.to(device)
            qw_e, s_e = torch_npu.npu_dynamic_mx_quant(
                w_e, dst_type=torch_npu.float8_e4m3fn
            )  # qw_e: [2I, H] float8, s_e: [2I, H//32] uint8
            # torch.stack doesn't support float8_e4m3fn on NPU; view as uint8 (same byte width) for stacking
            qw13_list.append(qw_e.view(torch.uint8))
            s13_list.append(s_e)

        qw2_list, s2_list = [], []
        for e in range(num_experts):
            w_e = layer.w2_weight.data[e]  # [H, I]
            if w_e.dtype not in (torch.float16, torch.bfloat16):
                w_e = w_e.to(torch.bfloat16)
            if not w_e.is_npu:
                w_e = w_e.to(device)
            qw_e, s_e = torch_npu.npu_dynamic_mx_quant(
                w_e, dst_type=torch_npu.float8_e4m3fn
            )  # qw_e: [H, I] float8, s_e: [H, I//32] uint8
            qw2_list.append(qw_e.view(torch.uint8))
            s2_list.append(s_e)

        # Stack as uint8 (float8_e4m3fn unsupported by NPU torch.stack), then view back to float8
        qw13 = torch.stack(qw13_list).view(torch.float8_e4m3fn)  # [E, 2I, H]
        s13 = torch.stack(s13_list)  # [E, 2I, H//32] uint8
        qw2 = torch.stack(qw2_list).view(torch.float8_e4m3fn)  # [E, H, I]
        s2 = torch.stack(s2_list)  # [E, H, I//32] uint8

        # Transpose for npu_grouped_matmul: [E, N, K] → [E, K, N] so K is dim-1.
        # No npu_format_cast — MXFP8 grouped_matmul does not require NZ format (unlike INT8 path).
        layer.w13_weight = Parameter(
            qw13.transpose(1, 2), requires_grad=False
        )  # [E, H, 2I] float8_e4m3fn
        layer.w13_weight_scale = Parameter(
            s13.transpose(1, 2), requires_grad=False
        )  # [E, H//32, 2I] uint8

        layer.w2_weight = Parameter(
            qw2.transpose(1, 2), requires_grad=False
        )  # [E, I, H] float8_e4m3fn
        layer.w2_weight_scale = Parameter(
            s2.transpose(1, 2), requires_grad=False
        )  # [E, I//32, H] uint8

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        from sglang.srt.layers.moe.token_dispatcher import (
            StandardCombineInput,
            StandardDispatchOutput,
        )

        if not isinstance(dispatch_output, StandardDispatchOutput):
            raise NotImplementedError(
                "NPUMXFP8FusedMoEMethod only supports FusedMoE (TP-only) dispatch. "
                "EPMoE variants (DeepEPMoE / NpuFuseEPMoE / MoriEPMoE) are not yet implemented."
            )

        hidden_states = dispatch_output.hidden_states
        topk_weights, topk_ids, _ = dispatch_output.topk_output
        topk_ids = topk_ids.to(torch.int32)
        topk_weights = topk_weights.to(hidden_states.dtype)

        output = npu_fused_experts_mxfp8(
            hidden_states=hidden_states,
            w13=layer.w13_weight,
            w13_scale=layer.w13_weight_scale,
            w2=layer.w2_weight,
            w2_scale=layer.w2_weight_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            top_k=topk_ids.shape[1],
        )
        return StandardCombineInput(hidden_states=output)
