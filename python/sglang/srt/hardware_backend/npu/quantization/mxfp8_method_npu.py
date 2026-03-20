from typing import Optional

import torch
from torch.nn.parameter import Parameter


class NPUMXFP8LinearMethod:
    """Ascend NPU MXFP8 weight processing and kernel calls.

    This class handles NPU-specific operations:
    - process_weights_after_loading: dtype casting and online quantization
    - apply: dynamic activation quantization + MXFP8 matmul

    Weight creation and config management are handled by
    MXFP8LinearAscendMethod in fp8.py.
    """

    MXFP8_BLOCK_SIZE = 32

    def process_weights_after_loading(
        self, layer: torch.nn.Module, is_serialized: bool
    ):
        import torch_npu

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

            if not weight_fp.is_npu:
                weight_fp = weight_fp.npu()

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
