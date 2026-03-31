"""ModelSlim MXFP4 scheme for pre-quantized weight inference on Ascend NPU.

Loads weights pre-quantized by msmodelslim and runs MXFP4 dual-level
matmul at inference via npu_dual_level_quant_matmul.

Checkpoint tensor formats (verified from msmodelslim export):
  weight:           [out, in]           float8_e4m3fn  (FP4 data in fp8 container)
  weight_scale:     [out, in/32]        uint8          (L0 block scales, e8m0+127)
  weight_dual_scale:[out, in/512, 1]    float32        (L1 coarse scales)

Reference: MindIE-SD W4A4MXFP4DualQuantLinear
(MindIE-SD/mindiesd/quantization/layer.py)
"""

from typing import List, Optional

import torch
import torch_npu

from sglang.multimodal_gen.runtime.models.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
)
from sglang.srt.layers.quantization.modelslim.schemes import ModelSlimLinearScheme

MXFP4_BLOCK_SIZE = 32
# L1 (dual) scale groups this many L0 blocks together.
# L1 block covers 16 * 32 = 512 elements.
MXFP4_DUAL_LEVEL_RATIO = 16


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
        # Each byte is a float8 container for FP4 data; the actual FP4 packing
        # (npu_dtype_cast → float4_e2m1fn_x2) happens in process_weights_after_loading.
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

        # L0 block scale: uint8 [out, in/32], e8m0 scale with +127 offset.
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

        # L1 (coarse) dual scale: float32 [out, in/512, 1].
        # Each L1 block covers 16 L0 blocks (512 elements).
        dual_scale_dim = scale_dim // MXFP4_DUAL_LEVEL_RATIO
        weight_dual_scale = GroupQuantScaleParameter(
            data=torch.empty(
                (output_size_per_partition, dual_scale_dim, 1),
                dtype=torch.float32,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_dual_scale", weight_dual_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module):
        # Cast weight from fp8 container to FP4 packed format
        weight = layer.weight.data
        if not weight.is_npu:
            weight = weight.to(f"npu:{torch.npu.current_device()}")
        weight = torch_npu.npu_dtype_cast(weight, torch_npu.float4_e2m1fn_x2)
        layer.weight = torch.nn.Parameter(weight, requires_grad=False)

        # Reshape weight_scale: [out, in/32] -> [out, in/64, 2]
        # The dual-level matmul API expects L0 scales in this 3D format
        weight_scale = layer.weight_scale.data
        weight_scale = weight_scale.reshape(weight_scale.shape[0], -1, 2)
        layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)

        # Transform weight_dual_scale: [out, in/512, 1] -> [in/512, out]
        weight_dual_scale = layer.weight_dual_scale.data
        weight_dual_scale = weight_dual_scale.squeeze(-1).transpose(0, 1).contiguous()
        layer.weight_dual_scale = torch.nn.Parameter(
            weight_dual_scale, requires_grad=False
        )

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

        # Flatten to 2D for npu_dynamic_dual_level_mx_quant
        input_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])

        # Dual-level MXFP4 activation quantization
        x1, l0_scale, l1_scale = torch_npu.npu_dynamic_dual_level_mx_quant(
            x_2d, smooth_scale=None
        )

        # Dual-level MXFP4 matmul
        output = torch_npu.npu_dual_level_quant_matmul(
            x1,
            layer.weight,
            l0_scale,
            layer.weight_dual_scale,
            l1_scale,
            layer.weight_scale,
            bias=bias.to(torch.float32) if bias is not None else None,
            output_dtype=original_dtype,
        )

        # Restore original shape
        output_shape = list(input_shape[:-1]) + [output.shape[-1]]
        return output.reshape(output_shape)
