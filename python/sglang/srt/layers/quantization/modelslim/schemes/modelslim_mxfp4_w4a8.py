"""ModelSlim W4A8_MXFP scheme for pre-quantized weight inference on Ascend NPU (SRT).

Loads weights pre-quantized by msmodelslim (FP4 weights packed as uint8, uint8
bias-shifted MXFP8_E8M0 scales) and runs W4A8 matmul at inference.

Weight format exported by msmodelslim (on_w4a8_mx_dynamic_per_block):
  weight:       pack_fp4_to_uint8 → uint8,  shape [out, in/2],    group_size=32
  weight_scale: (scale + 127).uint8         shape [out, in/32]

Inference:
  activation → npu_dynamic_mx_quant(float8_e4m3fn) → qx + per-token scale
  npu_quant_matmul(qx, weight, weight_scale, x1_dtype=fp8, x2_dtype=fp4)
"""

from typing import List, Optional

import torch
import torch_npu

from sglang.srt.layers.parameter import GroupQuantScaleParameter, ModelWeightParameter
from sglang.srt.layers.quantization.modelslim.schemes import ModelSlimLinearScheme

MXFP4_W4A8_BLOCK_SIZE = 32

_FLOAT8_E8M0FNU_DTYPE = getattr(
    torch_npu, "float8_e8m0fnu", getattr(torch, "float8_e8m0fnu", None)
)
_FLOAT4_E2M1FN_X2_DTYPE = getattr(
    torch_npu, "float4_e2m1fn_x2", getattr(torch, "float4_e2m1fn_x2", None)
)


class ModelSlimMXFP4W4A8Scheme(ModelSlimLinearScheme):

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

        # msmodelslim packs 2 FP4 values per uint8 → shape [out, in/2]
        weight = ModelWeightParameter(
            data=torch.empty(
                (output_size_per_partition, input_size_per_partition // 2),
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # msmodelslim exports weight_scale as uint8 with +127 bias, shape [out, in/32]
        scale_dim = input_size_per_partition // MXFP4_W4A8_BLOCK_SIZE
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
        # Same transform as ModelSlimMXFP8Scheme:
        # weight_scale: [out, in/32] → reshape [out, in/64, 2] → transpose [in/64, out, 2]
        # weight:       [out, in/2]  → transpose [in/2, out]
        n_dim, k_dim = layer.weight_scale.data.shape
        layer.weight_scale.data = layer.weight_scale.data.reshape(n_dim, k_dim // 2, 2)
        layer.weight.data = layer.weight.data.transpose(0, 1)
        layer.weight_scale.data = layer.weight_scale.data.transpose(0, 1)

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

        input_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])

        # Dynamically quantize activations to FP8 (A8 in W4A8)
        qx, input_scale = torch_npu.npu_dynamic_mx_quant(
            x_2d, dst_type=torch.float8_e4m3fn
        )

        # W4A8 matmul: FP8 activations × FP4 weights (already transposed at load time)
        output = torch_npu.npu_quant_matmul(
            qx,
            layer.weight,
            layer.weight_scale,
            scale_dtype=_FLOAT8_E8M0FNU_DTYPE,
            pertoken_scale=input_scale,
            pertoken_scale_dtype=_FLOAT8_E8M0FNU_DTYPE,
            bias=bias.to(torch.float32) if bias is not None else None,
            output_dtype=original_dtype,
            x1_dtype=torch.float8_e4m3fn,
            x2_dtype=_FLOAT4_E2M1FN_X2_DTYPE,
            group_sizes=[1, 1, MXFP4_W4A8_BLOCK_SIZE],
        )

        output_shape = list(input_shape[:-1]) + [output.shape[-1]]
        return output.reshape(output_shape)
