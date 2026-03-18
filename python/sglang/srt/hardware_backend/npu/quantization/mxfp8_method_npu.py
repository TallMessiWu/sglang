from typing import TYPE_CHECKING, List, Optional

import torch
from torch.nn.parameter import Parameter

from sglang.srt.layers.parameter import (
    BlockQuantScaleParameter,
    ModelWeightParameter,
)
from sglang.srt.layers.quantization.base_config import LinearMethodBase

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.fp8 import Fp8Config


class NPUMXFP8LinearMethod(LinearMethodBase):
    """Ascend NPU MXFP8 (Microscaling FP8) quantization for Linear layers.

    Supports two modes:
    - Online quantization: loads FP16/BF16 weights and quantizes them to MXFP8
      at weight loading time.
    - Offline quantization: loads pre-quantized FP8 weights with block scales
      from a serialized checkpoint.

    Uses torch_npu APIs:
    - npu_dynamic_mx_quant: dynamic MXFP8 activation quantization (block_size=32)
    - npu_quant_matmul: MXFP8 matrix multiplication with group_sizes=[1,1,32]
    """

    MXFP8_BLOCK_SIZE = 32

    def __init__(self, quant_config: "Fp8Config"):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        is_serialized = self.quant_config.is_checkpoint_fp8_serialized

        # Weight: fp8 if serialized checkpoint, else original dtype (will be
        # quantized in process_weights_after_loading)
        weight_dtype = torch.float8_e4m3fn if is_serialized else params_dtype
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=weight_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        if is_serialized:
            # Block scale: one scale per block of 32 elements along input dim.
            # Stored as uint8 (representing float8_e8m0fnu) in checkpoint.
            block_k = self.MXFP8_BLOCK_SIZE
            scale_cols = (input_size_per_partition + block_k - 1) // block_k
            scale = BlockQuantScaleParameter(
                data=torch.zeros(
                    output_size_per_partition,
                    scale_cols,
                    dtype=torch.uint8,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            )
            scale.format_ue8m0 = True
            layer.register_parameter("weight_scale_inv", scale)
        else:
            layer.register_parameter("weight_scale_inv", None)

    def process_weights_after_loading(self, layer: torch.nn.Module):
        import torch_npu

        is_serialized = self.quant_config.is_checkpoint_fp8_serialized

        if is_serialized:
            # Checkpoint already has fp8 weights + uint8 scales.
            # Ensure weight is float8_e4m3fn.
            if layer.weight.data.dtype != torch.float8_e4m3fn:
                layer.weight = Parameter(
                    torch_npu.npu_dtype_cast(
                        layer.weight.data, torch_npu.float8_e4m3fn
                    ),
                    requires_grad=False,
                )
            else:
                layer.weight.requires_grad_(False)

            # Scale is already uint8 (e8m0fnu), keep as-is.
            layer.weight_scale_inv.requires_grad_(False)
        else:
            # Online quantization: quantize FP16/BF16 weights to MXFP8.
            weight_fp = layer.weight.data
            if weight_fp.dtype not in (torch.float16, torch.bfloat16):
                weight_fp = weight_fp.to(torch.bfloat16)

            qw, w_scale = torch_npu.npu_dynamic_mx_quant(
                weight_fp, dst_type=torch_npu.float8_e4m3fn
            )
            layer.weight = Parameter(qw, requires_grad=False)
            layer.weight_scale_inv = Parameter(w_scale, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        import torch_npu

        original_dtype = x.dtype
        if original_dtype not in (torch.float16, torch.bfloat16):
            x = x.to(torch.bfloat16)
            original_dtype = torch.bfloat16

        # Dynamic MXFP8 activation quantization (block_size=32)
        qx, input_scale = torch_npu.npu_dynamic_mx_quant(
            x, dst_type=torch_npu.float8_e4m3fn
        )

        # MXFP8 quantized matmul
        output = torch_npu.npu_quant_matmul(
            qx,
            layer.weight.transpose(0, 1),
            layer.weight_scale_inv.transpose(0, 1),
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale=input_scale,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            bias=bias.to(torch.float32) if bias is not None else None,
            output_dtype=original_dtype,
            group_sizes=[1, 1, self.MXFP8_BLOCK_SIZE],
        )
        return output
