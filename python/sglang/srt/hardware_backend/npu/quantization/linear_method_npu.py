import logging
from typing import TYPE_CHECKING, Optional

import torch
from torch.nn.parameter import Parameter

from sglang.srt.hardware_backend.npu.utils import npu_format_cast
from sglang.srt.layers.quantization.base_config import LinearMethodBase

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig

logger = logging.getLogger(__name__)

MXFP8_BLOCK_SIZE = 32
# W4A8_MXFP block (group) size — fixed at 32 by the msmodelslim export format.
MXFP4_BLOCK_SIZE = 32


# NPU ops are reached via torch.ops.npu.* (registered when torch_npu is imported
# by the runtime), so this module needs no top-level `import torch_npu` and stays
# importable on CUDA/CPU/AMD/XPU CI.
def _get_float8_e8m0fnu_dtype():
    # Resolve lazily rather than as a module-level constant: this module is
    # imported early (during quant-scheme registration), so reading the dtype at
    # call time keeps it correct regardless of import order / platform.
    return getattr(torch, "float8_e8m0fnu", None)


class _NPULinearMethodBase(LinearMethodBase):

    def __init__(
        self,
        quant_config: Optional["QuantizationConfig"] = None,
    ):
        self.quant_config = quant_config


class NPUW8A8Int8LinearMethod(_NPULinearMethodBase):

    def process_weights_after_loading(self, layer: torch.nn.Module):
        layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight.data = npu_format_cast(layer.weight.data)

        layer.weight_scale.data = layer.weight_scale.data.flatten()
        # Compressed-tensors format doesn't have this field
        if hasattr(layer, "weight_offset"):
            layer.weight_offset.data = layer.weight_offset.data.flatten()

        expanding_factor = layer.weight.data.shape[0]
        layer.aclnn_input_scale = torch.nn.Parameter(
            layer.input_scale.data.repeat(expanding_factor).to(device="npu"),
            requires_grad=False,
        )
        layer.aclnn_input_scale_reciprocal = 1 / torch.nn.Parameter(
            layer.input_scale.data.repeat(expanding_factor).to(device="npu"),
            requires_grad=False,
        )
        layer.aclnn_input_offset = torch.nn.Parameter(
            layer.input_offset.data.repeat(expanding_factor).to(device="npu"),
            requires_grad=False,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from sglang.srt.layers.linear import RowParallelLinear

        original_dtype = x.dtype
        if original_dtype != torch.int8:
            x = torch.ops.npu.npu_quantize(
                x,
                layer.aclnn_input_scale_reciprocal,
                layer.aclnn_input_offset,
                torch.qint8,
                -1,
                False,
            )
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in Attention TP>1 case)
        if isinstance(layer, RowParallelLinear) and layer.tp_rank > 0:
            quant_bias = None
        else:
            quant_bias = layer.quant_bias
        return torch.ops.npu.npu_quant_matmul(
            x,
            layer.weight,
            layer.deq_scale,
            bias=quant_bias,
            output_dtype=original_dtype,
        )


class NPUW8A8Int8DynamicLinearMethod(_NPULinearMethodBase):

    def process_weights_after_loading(self, layer: torch.nn.Module):
        layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight.data = npu_format_cast(layer.weight.data)

        layer.weight_scale.data = layer.weight_scale.data.flatten()
        # Compressed-tensors format doesn't have this field
        if hasattr(layer, "weight_offset"):
            layer.weight_offset.data = layer.weight_offset.data.flatten()

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if isinstance(x, tuple):
            """dynamic_scale is calculated in malprolog kernel"""
            original_dtype = torch.bfloat16
            quant_out, dynamic_scale = x
        else:
            original_dtype = x.dtype
            quant_out, dynamic_scale = torch.ops.npu.npu_dynamic_quant(x)
        return torch.ops.npu.npu_quant_matmul(
            quant_out,
            layer.weight,
            layer.weight_scale,
            pertoken_scale=dynamic_scale.flatten(),
            bias=bias,
            output_dtype=original_dtype,
        )


class NPUMXFP8LinearMethod(_NPULinearMethodBase):
    """Ascend NPU MXFP8 linear method for LLM (SRT) models.

    Shared kernel for both the online config path (``--quantization mxfp8``) and
    the offline ModelSlimMXFP8Scheme (which delegates to this as ``self.kernel``).
    process_weights_after_loading branches on weight dtype: FP16/BF16 weights are
    quantised to MXFP8 at load time (online); pre-quantised float8_e4m3fn weights
    are only re-laid-out (offline). Inference: dynamic MXFP8 activation quant +
    MXFP8 matmul (block_size=32).
    """

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.parameter import ModelWeightParameter

        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        # Load weights in original dtype; quantise later in process_weights_after_loading
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data
        if weight.dtype == torch.float8_e4m3fn:
            # Offline (ModelSlim) path: weight is already MXFP8-quantised and
            # layer.weight_scale holds the uint8 block scales [out, in/32]. Only
            # re-layout to [in, out] / [in//64, out, 2] strided views below.
            n_dim, k_dim = layer.weight_scale.data.shape
            scale = layer.weight_scale.data.reshape(n_dim, k_dim // 2, 2)
            layer.weight = Parameter(weight.transpose(0, 1), requires_grad=False)
            layer.weight_scale_inv = Parameter(
                scale.transpose(0, 1), requires_grad=False
            )
            # weight_scale is now folded into weight_scale_inv (which keeps the
            # underlying storage alive via its view); drop the stale parameter so
            # it doesn't linger in named_parameters() / state_dict().
            del layer.weight_scale
        else:
            # Online path: quantise FP16/BF16 weights to MXFP8 at load time.
            if weight.dtype not in (torch.float16, torch.bfloat16):
                logger.warning(
                    "NPUMXFP8LinearMethod: weight dtype %s is not float16/bfloat16; "
                    "casting to bfloat16 before MXFP8 quantisation.",
                    weight.dtype,
                )
                weight = weight.to(torch.bfloat16)
            # Move weight to NPU if needed (cpu offload may move it back to CPU).
            if not weight.is_npu:
                weight = weight.to(f"npu:{torch.npu.current_device()}")
            # Online MXFP8 quantisation of weights (block_size=32).
            # qw: [out, in] float8_e4m3fn, w_scale: [out, in//64, 2] uint8.
            qw, w_scale = torch.ops.npu.npu_dynamic_mx_quant(
                weight, dst_type=torch.float8_e4m3fn
            )
            layer.weight = Parameter(qw.transpose(0, 1), requires_grad=False)
            layer.weight_scale_inv = Parameter(
                w_scale.transpose(0, 1), requires_grad=False
            )

        # Both paths produce weight [in, out] and weight_scale_inv [in//64, out,
        # 2] as strided transpose views — DO NOT call .contiguous(). The matmul
        # reduction loop scans the in-dim per output column; the [out, in]
        # row-major source gives stride-1 access for that scan via the transpose
        # view (matches msmodelslim's offline layout and vllm-ascend's
        # AscendW8A8MXFP8DynamicLinearMethod). Calling .contiguous() physically
        # reorders to [in, out] row-major, making the inner-loop stride = out and
        # tanking HBM bandwidth.

        # Cache FP32 bias once to avoid a per-forward dtype conversion + alloc.
        if (
            getattr(layer, "bias", None) is not None
            and layer.bias.dtype != torch.float32
        ):
            layer.bias_fp32 = Parameter(
                layer.bias.data.to(torch.float32), requires_grad=False
            )
        else:
            layer.bias_fp32 = None

    def apply(
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

        # Dynamic MXFP8 activation quantisation
        qx, input_scale = torch.ops.npu.npu_dynamic_mx_quant(
            x_2d, dst_type=torch.float8_e4m3fn
        )

        # MXFP8 matmul (weight & scale already transposed at load time)
        # Use the cached FP32 bias from process_weights_after_loading; fall back
        # to per-call conversion if the cache was bypassed (e.g. dynamic bias).
        if bias is None:
            quant_bias = None
        elif (
            bias is getattr(layer, "bias", None)
            and getattr(layer, "bias_fp32", None) is not None
        ):
            quant_bias = layer.bias_fp32
        else:
            quant_bias = bias.to(torch.float32)

        e8m0_dtype = _get_float8_e8m0fnu_dtype()
        output = torch.ops.npu.npu_quant_matmul(
            qx,
            layer.weight,
            layer.weight_scale_inv,
            scale_dtype=e8m0_dtype,
            pertoken_scale=input_scale,
            pertoken_scale_dtype=e8m0_dtype,
            bias=quant_bias,
            output_dtype=original_dtype,
            group_sizes=[1, 1, MXFP8_BLOCK_SIZE],
        )

        # Restore original shape (replace last dim with output features)
        output_shape = list(input_shape[:-1]) + [output.shape[-1]]
        return output.reshape(output_shape)


class NPU_W4A4DynamicLinearMethod(_NPULinearMethodBase):

    def process_weights_after_loading(self, layer):
        layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight_scale.data = layer.weight_scale.data.flatten()
        layer.weight_scale_fp32 = layer.weight_scale.data.to(torch.float32)
        layer.weight_offset.data = layer.weight_offset.data.flatten()
        layer.weight.data = torch.ops.npu.npu_convert_weight_to_int4pack(
            layer.weight.data.to(torch.int32)
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        tp_rank: Optional[int] = 0,
    ) -> torch.Tensor:
        original_dtype = x.dtype
        quant_out, dynamic_scale = torch.ops.npu.npu_dynamic_quant(
            x, dst_type=torch.quint4x2
        )
        return torch.ops.npu.npu_quant_matmul(
            quant_out,
            layer.weight,
            layer.weight_scale,
            pertoken_scale=dynamic_scale.flatten(),
            bias=bias,
            output_dtype=original_dtype,
        )


class NPUMXFP4W4A8LinearMethod(_NPULinearMethodBase):
    """Ascend NPU W4A8 online quantization: MXFP4 weights + MXFP8 activations.

    Weight quantization flow (process_weights_after_loading):
        BF16/FP16 weight → npu_dynamic_dual_level_mx_quant → FP4 + l0_scale(FP32) + l1_scale(FP8_E8M0)
        → npu_format_cast to FRACTAL_NZ (required by npu_dual_level_quant_matmul)
        → w_dual_scale transposed to [in/512, out] (required by matmul API)

    Inference flow (apply):
        FP16/BF16 activation → npu_dynamic_dual_level_mx_quant → FP4 + act_l0_scale + act_l1_scale
        → npu_dual_level_quant_matmul(FP4_act, FP4_weight, scales...) → FP16/BF16 output

    Note: The "A8" refers to the MXFP8 intermediate scale format (FP8_E8M0 l1_scale).
    The actual matmul compute is W4A4 (both operands in FP4) since there is no
    W4A8 mixed-precision kernel in the current torch_npu public API.

    Hardware requirement: Ascend 950 (Atlas A3). DualLevelQuantBatchMatmul is
    NOT supported on Atlas 800I A2/A3 or earlier chips.
    """

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.parameter import ModelWeightParameter

        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        # Load weights in original dtype; quantise to MXFP4 in
        # process_weights_after_loading.
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from sglang.srt.utils import get_npu_memory_capacity

        # Heuristic hardware check: npu_dynamic_dual_level_mx_quant requires
        # Ascend 950. Atlas A2/A3 have ≤64 GB per card; Ascend 950 has ≥96 GB.
        npu_mem_mb = get_npu_memory_capacity()
        if npu_mem_mb < 96 * 1024:
            logger.warning(
                "MXFP4 W4A8 dual-level quantization may not be supported on this "
                "hardware (detected NPU memory %.1f GB < 96 GB). "
                "npu_dynamic_dual_level_mx_quant requires Ascend 950 (Atlas A3). "
                "Continuing — expect a RuntimeError if the kernel is unavailable.",
                npu_mem_mb / 1024,
            )

        weight_fp = layer.weight.data
        if weight_fp.dtype not in (torch.float16, torch.bfloat16):
            weight_fp = weight_fp.to(torch.bfloat16)

        # Move to NPU if needed (cpu offload may have put it on CPU).
        if not weight_fp.is_npu:
            weight_fp = weight_fp.to(f"npu:{torch.npu.current_device()}")

        # Online MXFP4 dual-level quantisation of weights.
        # qw:          float4_e2m1fn_x2, shape [out, in]
        # w_dual_scale: float32,          shape [out, in/512, 1]  (L0)
        # w_scale:      float8_e8m0,      shape [out, (ceil(in/32)+1)//2, 2]  (L1)
        try:
            qw, w_dual_scale, w_scale = torch.ops.npu.npu_dynamic_dual_level_mx_quant(
                weight_fp, smooth_scale=None
            )
        except (RuntimeError, AttributeError) as e:
            raise RuntimeError(
                "npu_dynamic_dual_level_mx_quant failed — this operation requires "
                "Ascend 950 (Atlas A3). Atlas 800I A2/A3 and earlier chips do NOT "
                "support DualLevelQuantBatchMatmul. "
                f"Original error: {e}"
            ) from e

        # npu_dual_level_quant_matmul requires x2 in FRACTAL_NZ format (format=29);
        # view as int8 first because npu_format_cast only accepts int-dtype tensors.
        qw = torch.ops.npu.npu_format_cast(qw.view(torch.int8), 29)

        # npu_dual_level_quant_matmul expects x2_level0_scale shape [in/512, out]:
        # squeeze the trailing dim-1 axis, then transpose + contiguous.
        # NOTE: the strided-view (no .contiguous()) layout used by the MXFP8 dense
        # path regressed perf on Ascend 950/A3 for W4A8, so the contiguous copy is
        # kept here (perf-regression revert, 2026-06-16).
        w_dual_scale = w_dual_scale.squeeze(-1).transpose(0, 1).contiguous()

        layer.weight = Parameter(qw, requires_grad=False)
        layer.weight_dual_scale = Parameter(w_dual_scale, requires_grad=False)
        layer.weight_scale = Parameter(w_scale, requires_grad=False)
        # Cache FP32 bias once to avoid a per-forward dtype conversion + alloc.
        if (
            getattr(layer, "bias", None) is not None
            and layer.bias.dtype != torch.float32
        ):
            layer.bias_fp32 = Parameter(
                layer.bias.data.to(torch.float32), requires_grad=False
            )
        else:
            layer.bias_fp32 = None

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        original_dtype = x.dtype
        if original_dtype not in (torch.float16, torch.bfloat16):
            x = x.to(torch.bfloat16)
            original_dtype = torch.bfloat16

        # Flatten to 2D [tokens, hidden] for the dual-level quant API.
        input_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])

        # Dynamic MXFP4 activation quantisation (W4 activations → A4 for matmul).
        qx, act_l0_scale, act_l1_scale = torch.ops.npu.npu_dynamic_dual_level_mx_quant(
            x_2d, smooth_scale=None
        )

        # Use the cached FP32 bias from process_weights_after_loading; fall back
        # to per-call conversion if the cache was bypassed (e.g. dynamic bias).
        if bias is None:
            quant_bias = None
        elif (
            bias is getattr(layer, "bias", None)
            and getattr(layer, "bias_fp32", None) is not None
        ):
            quant_bias = layer.bias_fp32
        else:
            quant_bias = bias.to(torch.float32)

        # MXFP4 matmul: W4A4 compute (weight already in NZ format + transposed scales).
        output = torch.ops.npu.npu_dual_level_quant_matmul(
            qx,
            layer.weight,
            act_l0_scale,
            layer.weight_dual_scale,
            act_l1_scale,
            layer.weight_scale,
            bias=quant_bias,
            output_dtype=original_dtype,
        )

        # Restore original shape (replace last dim with output features).
        output_shape = list(input_shape[:-1]) + [output.shape[-1]]
        return output.reshape(output_shape)


class NPUMXFP4W4A8OfflineLinearMethod(_NPULinearMethodBase):
    """Ascend NPU offline W4A8 (ModelSlim ``W4A8_MXFP``): packed-FP4 weights + MXFP8 activations.

    Kernel for the offline ModelSlimMXFP4W4A8Scheme (delegated as ``self.kernel``).
    The msmodelslim ``W4A8_MXFP`` checkpoint stores weights as *packed FP4*
    (``pack_fp4_to_uint8`` → ``uint8`` shape ``[out, in//2]``) plus UE8M0 block
    scales (``uint8`` shape ``[out, in//group_size]``):

      process_weights_after_loading:
        weight (uint8 packed FP4 [out, in//2]) → npu_format_cast(29,
            customize_dtype=float8_e4m3fn, input_dtype=float4_e2m1fn_x2) → FRACTAL_NZ
            → transpose [in//2, out]
        weight_scale [out, in/32] → reshape [out, in/64, 2] → transpose → [in/64, out, 2]

      apply:
        BF16/FP16 activation → npu_dynamic_mx_quant(dst=float8_e4m3fn)  (A8, MXFP8)
        → npu_quant_matmul(x2_dtype=float4_e2m1fn_x2, group_sizes=[0, 0, block])

    Mirrors vllm-ascend ``AscendW4A8MXFPDynamicLinearMethod`` exactly (Ascend 950/A5).
    The weight is cast to FRACTAL_NZ then transposed; ``npu_dynamic_mx_quant`` already
    returns a 3D ``[tokens, in//64, 2]`` block scale so the matmul needs no extra
    scale-layout normalization.

    ⚠️ REQUIRES a recent torch_npu build for the FP4 ``npu_quant_matmul``. On the
    A5 this device forces ``allow_internal_format=False`` (the NZ cast still produces
    a ``FRACTAL_NZ_C0_16`` tensor). Older torch_npu (e.g. ``2.10.0.dev20260320``)
    had a broken FP4 matmul that rejected the NZ weight ("x2 ... it is 2") or
    segfaulted in ``atb::OperationSetup``; ``2.10.0.post1.dev20260624`` (and later)
    runs the vllm-aligned NZ path correctly. If you hit those errors, update
    torch_npu — do NOT "fix" it by switching the weight to ND.

    Unlike the *online* ``NPUMXFP4W4A8LinearMethod`` (dual-level MXFP4, W4A4 compute
    via ``npu_dual_level_quant_matmul``), this offline path is a true W4(weight)
    A8(activation) single-level matmul. ``group_size`` is fixed at 32 by the
    ``W4A8_MXFP`` export format.
    """

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Mirror vllm-ascend AscendW4A8MXFPDynamicLinearMethod: cast the packed-FP4
        # weight to FRACTAL_NZ then transpose. npu_format_cast needs the FP4-unpack
        # kwargs the sglang util wrapper doesn't expose, hence the runtime
        # torch_npu import (NPU-only). Requires a recent torch_npu (see class
        # docstring): older builds reject the NZ weight ("x2 ... it is 2").
        import torch_npu

        # weight: packed-FP4 uint8 [out, in//2] -> FRACTAL_NZ (float8_e4m3fn view)
        # -> transpose to [in//2, out].
        layer.weight.data = torch_npu.npu_format_cast(
            layer.weight.data,
            29,
            customize_dtype=torch.float8_e4m3fn,
            input_dtype=torch_npu.float4_e2m1fn_x2,
        )
        layer.weight.data = layer.weight.data.transpose(-1, -2)
        # weight_scale: [out, in/32] uint8 -> [in/64, out, 2].
        n, k = layer.weight_scale.data.shape
        layer.weight_scale.data = layer.weight_scale.data.reshape(
            n, k // 2, 2
        ).transpose(-3, -2)

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

        # Flatten to 2D [tokens, hidden] for npu_dynamic_mx_quant.
        input_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])

        # Dynamic MXFP8 activation quantisation (A8).
        quantized_x, dynamic_scale = torch_npu.npu_dynamic_mx_quant(
            x_2d, dst_type=torch.float8_e4m3fn
        )

        if bias is not None and bias.dtype != torch.float32:
            bias = bias.to(torch.float32)

        # [DEBUG-W4A8] temporary instrumentation: dump the real matmul operands so the
        # segfaulting layer's actual format/shape/strides can be compared against the
        # standalone diagnostic. Remove once the offline W4A8 e2e is fixed.
        def _dbg(t):
            if t is None:
                return "None"
            try:
                f = torch_npu.get_npu_format(t)
            except Exception as e:  # noqa: BLE001
                f = f"<err {e}>"
            return (
                f"fmt={f} shape={tuple(t.shape)} dtype={t.dtype} "
                f"contig={t.is_contiguous()} stride={tuple(t.stride())}"
            )

        print(f"[DEBUG-W4A8 apply] prefix={getattr(layer, 'prefix', '?')}", flush=True)
        print("  layer.weight      :", _dbg(layer.weight), flush=True)
        print("  layer.weight.data :", _dbg(layer.weight.data), flush=True)
        print("  layer.weight_scale:", _dbg(layer.weight_scale), flush=True)
        print("  quantized_x       :", _dbg(quantized_x), flush=True)
        print("  dynamic_scale     :", _dbg(dynamic_scale), flush=True)
        print("  bias              :", _dbg(bias), flush=True)

        # W4(weight)A8(activation) matmul, mirroring vllm-ascend exactly.
        output = torch_npu.npu_quant_matmul(
            quantized_x,
            layer.weight,
            layer.weight_scale,
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale=dynamic_scale,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            bias=bias,
            output_dtype=original_dtype,
            x2_dtype=torch_npu.float4_e2m1fn_x2,
            group_sizes=[0, 0, MXFP4_BLOCK_SIZE],
        )

        # [DEBUG-W4A8] force a device sync so an async kernel error surfaces HERE,
        # pinned to the matmul whose operands were just printed above, instead of
        # leaking out as a misattributed segfault at a later sync point. Remove
        # together with the dump instrumentation once the e2e is fixed.
        torch.npu.synchronize()
        print("  -> matmul OK", flush=True)

        # Restore original shape (replace last dim with output features).
        output_shape = list(input_shape[:-1]) + [output.shape[-1]]
        return output.reshape(output_shape)
