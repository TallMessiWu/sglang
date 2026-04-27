"""ModelSlim MXFP4 (W4A4) scheme for pre-quantized weight inference on Ascend NPU (SRT).

Loads weights pre-quantized by msmodelslim and runs single-level MXFP4
matmul at inference via npu_quant_matmul (group_sizes=[1, 1, 32]).

Checkpoint tensor formats (from msmodelslim export):
  weight:       [out, in]      float8_e4m3fn  (FP4 data in fp8 container, 1 FP4 per byte)
  weight_scale: [out, in/32]   uint8          (E8M0 block scales)

process_weights_after_loading:
  1. Cast weight float8_e4m3fn -> float4_e2m1fn_x2 (2 FP4 per byte), shape [out, in/2]
  2. Transpose weight [out, in/2] -> [in/2, out]
  3. Transpose weight_scale [out, in/32] -> [in/32, out]

Inference (apply_weights):
  1. Dynamic single-level MXFP4 activation quantization via npu_dynamic_mx_quant
  2. npu_quant_matmul with group_sizes=[1, 1, 32]

Mirrors the online NPUSingleLevelMXFP4LinearMethod but loads pre-quantized weights
instead of quantizing from BF16/FP16 at load time.
"""

from typing import List, Optional

import torch
import torch_npu

from sglang.srt.layers.parameter import GroupQuantScaleParameter, ModelWeightParameter
from sglang.srt.layers.quantization.modelslim.schemes import ModelSlimLinearScheme

MXFP4_BLOCK_SIZE = 32

_FLOAT4_E2M1FN_X2_DTYPE = getattr(
    torch_npu, "float4_e2m1fn_x2", getattr(torch, "float4_e2m1fn_x2", None)
)
_FLOAT8_E8M0FNU_DTYPE = getattr(
    torch_npu, "float8_e8m0fnu", getattr(torch, "float8_e8m0fnu", None)
)


class ModelSlimMXFP4Scheme(ModelSlimLinearScheme):

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
        weight_loader = extra_weight_attrs.get("weight_loader")
        output_size_per_partition = sum(output_partition_sizes)

        # msmodelslim exports weight as float8_e4m3fn, shape [out, in].
        # Each byte carries one FP4 value (fp8 container); npu_dtype_cast to
        # float4_e2m1fn_x2 packs 2 FP4 values per byte in process_weights_after_loading.
        weight = ModelWeightParameter(
            data=torch.empty(
                (output_size_per_partition, input_size_per_partition),
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # msmodelslim exports weight_scale as uint8 (E8M0), shape [out, in/32].
        scale_dim = input_size_per_partition // MXFP4_BLOCK_SIZE
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                (output_size_per_partition, scale_dim),
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module):
        weight = layer.weight.data
        if not weight.is_npu:
            weight = weight.to(f"npu:{torch.npu.current_device()}")

        # Cast from fp8 container to float4_e2m1fn_x2 (2 FP4 per byte).
        # Shape changes: [out, in] -> [out, in/2].
        weight_fp4 = torch_npu.npu_dtype_cast(weight, _FLOAT4_E2M1FN_X2_DTYPE)

        # Transpose to [in/2, out] for npu_quant_matmul.
        # Use .data assignment (not Parameter(..., contiguous)) to preserve the
        # non-contiguous transpose view — npu_quant_matmul reads strides directly
        # and .contiguous() would reorder data, breaking block-scale alignment.
        layer.weight = torch.nn.Parameter(
            weight_fp4.transpose(0, 1), requires_grad=False
        )

        weight_scale = layer.weight_scale.data
        if not weight_scale.is_npu:
            weight_scale = weight_scale.to(f"npu:{torch.npu.current_device()}")
        # Transpose weight_scale [out, in/32] -> [in/32, out].
        # Avoid .contiguous() to keep block-scale mapping intact.
        layer.weight_scale.data = weight_scale.transpose(0, 1)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        original_dtype = x.dtype
        if original_dtype not in (torch.float16, torch.bfloat16):
            x = x.to(torch.bfloat16)
            original_dtype = torch.bfloat16

        # Flatten to 2D [tokens, hidden] for npu_dynamic_mx_quant
        input_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])

        # Dynamic single-level MXFP4 activation quantization
        qx, input_scale = torch_npu.npu_dynamic_mx_quant(
            x_2d, dst_type=_FLOAT4_E2M1FN_X2_DTYPE, round_mode="round"
        )

        # Single-level MXFP4 matmul (weight & scale already transposed at load time)
        output = torch_npu.npu_quant_matmul(
            qx,
            layer.weight,
            layer.weight_scale,
            scale_dtype=_FLOAT8_E8M0FNU_DTYPE,
            pertoken_scale=input_scale,
            pertoken_scale_dtype=_FLOAT8_E8M0FNU_DTYPE,
            bias=bias.to(torch.float32) if bias is not None else None,
            output_dtype=original_dtype,
            x1_dtype=_FLOAT4_E2M1FN_X2_DTYPE,
            x2_dtype=_FLOAT4_E2M1FN_X2_DTYPE,
            group_sizes=[1, 1, MXFP4_BLOCK_SIZE],
        )

        # Restore original shape
        output_shape = list(input_shape[:-1]) + [output.shape[-1]]
        return output.reshape(output_shape)
