"""ModelSlim W4A8_MXFP scheme for pre-quantized weight inference on Ascend NPU (SRT).

The current msmodelslim ``W4A8_MXFP`` checkpoint stores weights in the *same*
layout as ``W8A8_MXFP8``:

    weight:       float8_e4m3fn, shape [out, in],    group_size=32
    weight_scale: uint8 (+127 biased),  shape [out, in/32]

so weight creation, weight post-processing and the forward pass are all
identical to :class:`ModelSlimMXFP8Scheme` (which delegates to
``NPUMXFP8LinearMethod``). The W4A8_MXFP vs W8A8_MXFP8 distinction lives in how
msmodelslim derives the values offline, not in the inference path — hence this
scheme reuses the MXFP8 one wholesale.

A future packed-FP4 checkpoint format (``pack_fp4_to_uint8`` → uint8 shape
``[out, in/2]``) would no longer match this layout and would need its own scheme.
"""

from sglang.srt.layers.quantization.modelslim.schemes.modelslim_mxfp8 import (
    ModelSlimMXFP8Scheme,
)


class ModelSlimMXFP4W4A8Scheme(ModelSlimMXFP8Scheme):
    """W4A8_MXFP offline scheme — identical layout/inference to MXFP8."""
