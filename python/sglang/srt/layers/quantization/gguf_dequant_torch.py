# SPDX-License-Identifier: Apache-2.0
"""Pure-PyTorch vectorized GGML dequantization for non-CUDA/MUSA backends."""

from __future__ import annotations

import gguf
import torch
from gguf import GGMLQuantizationType as WeightType

TORCH_DEQUANT_TYPES: frozenset[WeightType] = frozenset(
    {
        WeightType.Q4_0,
        WeightType.Q4_1,
        WeightType.Q5_0,
        WeightType.Q5_1,
        WeightType.Q8_0,
        WeightType.Q2_K,
        WeightType.Q3_K,
        WeightType.Q4_K,
        WeightType.Q5_K,
        WeightType.Q6_K,
    }
)


def _decode_fp16_scale(raw_bytes: torch.Tensor) -> torch.Tensor:
    """Decode an unaligned 2-byte fp16 slice into float32."""
    return raw_bytes.contiguous().view(torch.float16).to(torch.float32)


def _get_scale_min_k4(scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack 6-bit scales and mins from 12-byte Q4_K / Q5_K scale headers."""
    scales3 = scales.view(-1, 3, 4)
    s0, s1, s2 = scales3[:, 0], scales3[:, 1], scales3[:, 2]
    sc = torch.cat([s0 & 0x3F, (s2 & 0x0F) | ((s0 >> 2) & 0x30)], dim=-1)
    m = torch.cat([s1 & 0x3F, (s2 >> 4) | ((s1 >> 2) & 0x30)], dim=-1)
    return sc, m


def _unpack_planar_nibbles(qs: torch.Tensor) -> torch.Tensor:
    """Unpack low nibbles followed by high nibbles across each block."""
    return torch.cat([qs & 0x0F, qs >> 4], dim=-1)


def _unpack_k4_nibbles(qs: torch.Tensor) -> torch.Tensor:
    """Unpack 128 bytes of K-quant nibbles into 8 sub-blocks of 32 values."""
    qs4 = qs.view(-1, 4, 32)
    return torch.stack([qs4 & 0x0F, qs4 >> 4], dim=2).reshape(-1, 8, 32)


def _dequantize_q4_0(blocks: torch.Tensor) -> torch.Tensor:
    d = _decode_fp16_scale(blocks[:, :2])
    q = _unpack_planar_nibbles(blocks[:, 2:]).to(torch.float32) - 8.0
    return d * q


def _dequantize_q4_1(blocks: torch.Tensor) -> torch.Tensor:
    d = _decode_fp16_scale(blocks[:, :2])
    m = _decode_fp16_scale(blocks[:, 2:4])
    q = _unpack_planar_nibbles(blocks[:, 4:]).to(torch.float32)
    return d * q + m


def _dequantize_q5_0(blocks: torch.Tensor) -> torch.Tensor:
    d = _decode_fp16_scale(blocks[:, :2])
    shifts = torch.arange(8, dtype=torch.uint8, device=blocks.device).view(1, 1, 8)
    qh = ((blocks[:, 2:6].unsqueeze(-1) >> shifts) & 0x01).reshape(-1, 32)
    ql = _unpack_planar_nibbles(blocks[:, 6:])
    q = (ql | (qh << 4)).to(torch.float32) - 16.0
    return d * q


def _dequantize_q5_1(blocks: torch.Tensor) -> torch.Tensor:
    d = _decode_fp16_scale(blocks[:, :2])
    m = _decode_fp16_scale(blocks[:, 2:4])
    shifts = torch.arange(8, dtype=torch.uint8, device=blocks.device).view(1, 1, 8)
    qh = ((blocks[:, 4:8].unsqueeze(-1) >> shifts) & 0x01).reshape(-1, 32)
    ql = _unpack_planar_nibbles(blocks[:, 8:])
    q = (ql | (qh << 4)).to(torch.float32)
    return d * q + m


def _dequantize_q8_0(blocks: torch.Tensor) -> torch.Tensor:
    d = _decode_fp16_scale(blocks[:, :2])
    qs = blocks[:, 2:].view(torch.int8).to(torch.float32)
    return d * qs


def _dequantize_q2_k(blocks: torch.Tensor) -> torch.Tensor:
    scales = blocks[:, :16]
    qs = blocks[:, 16:80]
    d = _decode_fp16_scale(blocks[:, 80:82])
    dmin = _decode_fp16_scale(blocks[:, 82:84])
    dl = (d * (scales & 0x0F).to(torch.float32)).unsqueeze(-1)
    ml = (dmin * (scales >> 4).to(torch.float32)).unsqueeze(-1)
    shifts = torch.tensor([0, 2, 4, 6], dtype=torch.uint8, device=blocks.device).view(
        1, 1, 4, 1
    )
    q = ((qs.view(-1, 2, 1, 32) >> shifts) & 0x03).reshape(-1, 16, 16).to(torch.float32)
    return (dl * q - ml).reshape(-1, 256)


def _dequantize_q3_k(blocks: torch.Tensor) -> torch.Tensor:
    hmask = blocks[:, :32]
    qs = blocks[:, 32:96]
    scales = blocks[:, 96:108]
    d = _decode_fp16_scale(blocks[:, 108:110])

    lscales = torch.cat([scales[:, :8] & 0x0F, scales[:, :8] >> 4], dim=-1)
    shifts4 = torch.tensor([0, 2, 4, 6], dtype=torch.uint8, device=blocks.device).view(
        1, 4, 1
    )
    hscales = ((scales[:, 8:12].unsqueeze(1) >> shifts4) & 0x03).reshape(-1, 16)
    sc = (lscales | (hscales << 4)).to(torch.float32) - 32.0
    dl = (d * sc).unsqueeze(-1)

    ql = ((qs.view(-1, 2, 1, 32) >> shifts4.view(1, 1, 4, 1)) & 0x03).reshape(
        -1, 16, 16
    )
    shifts8 = torch.arange(8, dtype=torch.uint8, device=blocks.device).view(1, 1, 8, 1)
    qh = (((hmask.view(-1, 1, 1, 32) >> shifts8) & 0x01) ^ 0x01).reshape(-1, 16, 16)
    q = ql.to(torch.float32) - (qh << 2).to(torch.float32)
    return (dl * q).reshape(-1, 256)


def _dequantize_q4_k(blocks: torch.Tensor) -> torch.Tensor:
    d = _decode_fp16_scale(blocks[:, :2])
    dmin = _decode_fp16_scale(blocks[:, 2:4])
    sc, m = _get_scale_min_k4(blocks[:, 4:16])
    dl = (d * sc.to(torch.float32)).unsqueeze(-1)
    dm = (dmin * m.to(torch.float32)).unsqueeze(-1)
    q = _unpack_k4_nibbles(blocks[:, 16:144]).to(torch.float32)
    return (dl * q - dm).reshape(-1, 256)


def _dequantize_q5_k(blocks: torch.Tensor) -> torch.Tensor:
    d = _decode_fp16_scale(blocks[:, :2])
    dmin = _decode_fp16_scale(blocks[:, 2:4])
    sc, m = _get_scale_min_k4(blocks[:, 4:16])
    dl = (d * sc.to(torch.float32)).unsqueeze(-1)
    dm = (dmin * m.to(torch.float32)).unsqueeze(-1)
    shifts8 = torch.arange(8, dtype=torch.uint8, device=blocks.device).view(1, 8, 1)
    qh = (blocks[:, 16:48].unsqueeze(1) >> shifts8) & 0x01
    ql = _unpack_k4_nibbles(blocks[:, 48:176])
    q = (ql | (qh << 4)).to(torch.float32)
    return (dl * q - dm).reshape(-1, 256)


def _dequantize_q6_k(blocks: torch.Tensor) -> torch.Tensor:
    ql2 = blocks[:, :128].view(-1, 2, 64)
    ql = torch.stack([ql2 & 0x0F, ql2 >> 4], dim=2).reshape(-1, 8, 32)
    qh2 = blocks[:, 128:192].view(-1, 2, 32)
    shifts = torch.tensor([0, 2, 4, 6], dtype=torch.uint8, device=blocks.device).view(
        1, 1, 4, 1
    )
    qh = ((qh2.unsqueeze(2) >> shifts) & 0x03).reshape(-1, 8, 32)
    q = ((ql | (qh << 4)).to(torch.float32) - 32.0).reshape(-1, 16, 16)

    sc = blocks[:, 192:208].view(torch.int8).to(torch.float32)
    d = _decode_fp16_scale(blocks[:, 208:210])
    dl = (d * sc).unsqueeze(-1)
    return (dl * q).reshape(-1, 256)


_DEQUANTIZERS = {
    WeightType.Q4_0: _dequantize_q4_0,
    WeightType.Q4_1: _dequantize_q4_1,
    WeightType.Q5_0: _dequantize_q5_0,
    WeightType.Q5_1: _dequantize_q5_1,
    WeightType.Q8_0: _dequantize_q8_0,
    WeightType.Q2_K: _dequantize_q2_k,
    WeightType.Q3_K: _dequantize_q3_k,
    WeightType.Q4_K: _dequantize_q4_k,
    WeightType.Q5_K: _dequantize_q5_k,
    WeightType.Q6_K: _dequantize_q6_k,
}


def ggml_dequantize_torch(
    qweight: torch.Tensor,
    qweight_type: int,
    m: int,
    n: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Dequantize a packed GGML tensor using vectorized PyTorch ops."""
    if m <= 0 or n <= 0:
        raise ValueError(f"Expected positive dimensions (m, n), got ({m}, {n})")
    try:
        weight_type = WeightType(int(qweight_type))
    except ValueError as exc:
        raise NotImplementedError(
            f"Unsupported GGUF quantization type: {qweight_type}"
        ) from exc

    dequantize_fn = _DEQUANTIZERS.get(weight_type)
    if dequantize_fn is None:
        raise NotImplementedError(
            f"GGUF quantization type {weight_type.name} is not supported by "
            f"the PyTorch dequantizer on {qweight.device.type}"
        )

    if qweight.dtype != torch.uint8:
        raise TypeError(f"Expected uint8 packed GGUF tensor, got {qweight.dtype}")

    block_size, type_size = gguf.GGML_QUANT_SIZES[weight_type]
    total_elements = m * n
    if total_elements % block_size != 0:
        raise ValueError(
            f"GGUF output element count {total_elements} is not a multiple of "
            f"{weight_type.name} block size {block_size}"
        )
    expected_bytes = (total_elements // block_size) * type_size
    if qweight.numel() != expected_bytes:
        raise ValueError(
            f"GGUF packed tensor has {qweight.numel()} bytes, expected "
            f"{expected_bytes} bytes for shape ({m}, {n}) and {weight_type.name}"
        )

    blocks = qweight.contiguous().view(-1, type_size)
    out = dequantize_fn(blocks)
    target_dtype = torch.float16 if dtype is None else dtype
    return out.to(dtype=target_dtype).view(m, n)


__all__ = ["TORCH_DEQUANT_TYPES", "ggml_dequantize_torch"]
