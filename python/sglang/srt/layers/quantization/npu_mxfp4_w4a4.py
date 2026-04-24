"""Ascend NPU MXFP4 W4A4 online quantization config.

Triggered by ``--quantization mxfp4_w4a4_npu``.

Online mode: loads FP16/BF16 weights, quantises to single-level MXFP4 in
``process_weights_after_loading``.  During inference, activations are
dynamically quantised to MXFP4 and ``npu_quant_matmul`` is used with
group_sizes=[1, 1, MXFP4_BLOCK_SIZE].

Analogous to the MXFP8 online quant path (linear_method_npu.NPUMXFP8LinearMethod)
but using FP4 dtype (float4_e2m1fn) for both weights and activations.

Hardware requirement: verify float4_e2m1fn support on target Ascend hardware.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

import torch

from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.unquant import (
    UnquantizedFusedMoEMethod,
    UnquantizedLinearMethod,
)
from sglang.srt.layers.quantization.utils import is_layer_skipped

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class NPUMxfp4W4A4Config(QuantizationConfig):
    """Quantization config for Ascend NPU single-level MXFP4 W4A4 online quantization.

    Weights are quantised online to single-level MXFP4 format (float4_e2m1fn)
    during model loading.  Activations are quantised dynamically to MXFP4 at
    inference time.  The matmul is executed via ``torch_npu.npu_quant_matmul``
    with group_sizes=[1, 1, MXFP4_BLOCK_SIZE].
    """

    def __init__(
        self,
        ignored_layers: Optional[List[str]] = None,
        packed_modules_mapping: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        self.ignored_layers = ignored_layers or []
        self.packed_modules_mapping = packed_modules_mapping or {}

    @classmethod
    def get_name(cls) -> str:
        return "mxfp4_w4a4_npu"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0  # NPU bypasses CUDA capability checks

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict) -> "NPUMxfp4W4A4Config":
        ignored_layers = cls.get_from_keys_or(
            config, ["ignored_layers", "modules_to_not_convert"], None
        )
        if ignored_layers:
            normalized: List[str] = []
            for layer in ignored_layers:
                base = layer.removeprefix("model.")
                normalized.append(base)
                normalized.append(f"model.{base}")
            ignored_layers = normalized
        packed_modules_mapping = (
            cls.get_from_keys_or(config, ["packed_modules_mapping"], {}) or {}
        )
        return cls(
            ignored_layers=ignored_layers,
            packed_modules_mapping=packed_modules_mapping,
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix,
                self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
                NPUSingleLevelMXFP4LinearMethod,
            )

            return NPUSingleLevelMXFP4LinearMethod(self)
        elif isinstance(layer, FusedMoE):
            # MoE single-level MXFP4 W4A4 not yet implemented; fall back to unquantised
            logger.warning(
                "MXFP4 W4A4 quantization is not yet supported for FusedMoE layers "
                "(prefix=%s). Falling back to unquantized MoE — MoE weights will "
                "run in full precision (BF16/FP16).",
                prefix,
            )
            return UnquantizedFusedMoEMethod(
                layer.use_triton_kernels, layer.use_flashinfer_trtllm_moe
            )
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []
