# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Optional

import torch

from sglang.srt.layers.parameter import (
    ChannelQuantScaleParameter,
    ModelWeightParameter,
)
from sglang.srt.layers.quantization.modelslim.schemes.modelslim_scheme import ModelSlimLinearScheme


class ModelSlimMXFP8(ModelSlimLinearScheme):
    """msmodelslim W8A8_MXFP8 quantization scheme for Ascend NPU.

    References MindIE-SD's W8A8MXFP8QuantLinear implementation.
    Uses torch_npu.npu_dynamic_mx_quant for activation quantization
    and torch_npu.npu_quant_matmul with group_sizes=[1,1,32] for MXFP8 matmul.
    """

    def __init__(
        self,
        quant_config: Dict[str, any],
        prefix: str,
    ):
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
        weight_loader = extra_weight_attrs.get("weight_loader")
        output_size_per_partition = sum(output_partition_sizes)

        # Weight: stored as int8 in safetensors, cast to float8_e4m3fn at runtime
        weight = ModelWeightParameter(
            data=torch.empty(
                (output_size_per_partition, input_size_per_partition),
                dtype=torch.int8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # Weight scale: float8_e8m0fnu stored as uint8 in safetensors
        # msmodelslim export format: [out_features, in_features // 32 * 2]
        # Will be reshaped to [out_features, in_features // 32, 2] after loading
        scale_cols = (input_size_per_partition + 31) // 32 * 2
        weight_scale = ChannelQuantScaleParameter(
            data=torch.empty(
                (output_size_per_partition, scale_cols),
                dtype=torch.uint8,
            ),
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module):
        import torch_npu

        # Cast weight from int8 to float8_e4m3fn
        weight_data = layer.weight.data
        layer.weight = torch.nn.Parameter(
            torch_npu.npu_dtype_cast(weight_data, torch_npu.float8_e4m3fn),
            requires_grad=False,
        )

        # Reshape weight_scale: [out, cols] -> [out, cols // 2, 2]
        ws = layer.weight_scale.data
        layer.weight_scale = torch.nn.Parameter(
            ws.reshape(ws.shape[0], -1, 2),
            requires_grad=False,
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        import torch_npu

        original_dtype = x.dtype
        if x.dtype not in (torch.float16, torch.bfloat16):
            x = x.to(torch.bfloat16)

        # Dynamic MXFP8 activation quantization
        qx, input_scale = torch_npu.npu_dynamic_mx_quant(
            x, dst_type=torch_npu.float8_e4m3fn
        )

        # Weight dtype cast + transpose (following MindIE-SD pattern)
        w = layer.weight
        if w.dtype != torch.float8_e4m3fn:
            w = torch_npu.npu_dtype_cast(w, torch_npu.float8_e4m3fn)
        w = w.transpose(0, 1)

        # Prepare bias: npu_quant_matmul requires float32 bias
        matmul_bias = None
        if bias is not None:
            matmul_bias = bias.to(torch.float32)

        # MXFP8 matrix multiplication
        output = torch_npu.npu_quant_matmul(
            qx,
            w,
            layer.weight_scale.transpose(0, 1),
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale=input_scale,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            bias=matmul_bias,
            output_dtype=original_dtype,
            group_sizes=[1, 1, 32],
        )
        return output
