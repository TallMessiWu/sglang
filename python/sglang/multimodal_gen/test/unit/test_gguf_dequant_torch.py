# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for pure-PyTorch GGML dequantization."""

import unittest

import gguf
import torch
import torch.nn.functional as F
from gguf import GGMLQuantizationType as WeightType

from sglang.multimodal_gen.runtime.layers.linear import ReplicatedLinear
from sglang.multimodal_gen.runtime.layers.quantization.gguf import (
    GGUFConfig,
    GGUFEmbeddingMethod,
    GGUFLinearMethod,
)
from sglang.multimodal_gen.runtime.loader.gguf_weights import GGUFTensorMeta
from sglang.srt.layers.quantization.gguf import (
    apply_gguf_embedding,
    dequantize_gguf_weight,
)
from sglang.srt.layers.quantization.gguf_dequant_torch import (
    TORCH_DEQUANT_TYPES,
    ggml_dequantize_torch,
)

# Byte offsets of fp16 scale/min headers inside each GGML block.
_FP16_FIELD_OFFSETS: dict[WeightType, tuple[int, ...]] = {
    WeightType.Q4_0: (0,),
    WeightType.Q4_1: (0, 2),
    WeightType.Q5_0: (0,),
    WeightType.Q5_1: (0, 2),
    WeightType.Q8_0: (0,),
    WeightType.Q2_K: (80, 82),
    WeightType.Q3_K: (108,),
    WeightType.Q4_K: (0, 2),
    WeightType.Q5_K: (0, 2),
    WeightType.Q6_K: (208,),
}


def _make_packed_blocks(
    weight_type: WeightType, rows: int, cols: int, seed: int = 17
) -> torch.Tensor:
    """Create deterministic non-symmetric packed blocks with finite fp16 scales."""
    block_size, type_size = gguf.GGML_QUANT_SIZES[weight_type]
    assert cols % block_size == 0
    n_blocks = (rows * cols) // block_size

    gen = torch.Generator().manual_seed(seed + int(weight_type))
    blocks = torch.randint(
        0, 256, (n_blocks, type_size), generator=gen, dtype=torch.uint8
    )
    scales = torch.tensor(
        [0.0625, -0.125, 0.1875, -0.25, 0.3125, -0.375, 0.5, -0.75],
        dtype=torch.float16,
    )
    for field_idx, offset in enumerate(_FP16_FIELD_OFFSETS[weight_type]):
        idx = (torch.arange(n_blocks) + field_idx * 3) % len(scales)
        field_bytes = scales[idx].view(torch.uint8).reshape(n_blocks, 2)
        blocks[:, offset : offset + 2] = field_bytes
    return blocks.reshape(rows, (cols // block_size) * type_size)


class TestGGMLDequantizeTorch(unittest.TestCase):
    def test_all_supported_types_match_gguf_reference(self):
        self.assertEqual(set(_FP16_FIELD_OFFSETS), set(TORCH_DEQUANT_TYPES))
        rows, cols = 6, 512
        for weight_type in sorted(TORCH_DEQUANT_TYPES, key=int):
            with self.subTest(weight_type=weight_type.name):
                packed = _make_packed_blocks(weight_type, rows, cols)
                ref_np = gguf.dequantize(packed.numpy(), weight_type)
                ref_fp32 = torch.from_numpy(ref_np.copy()).to(torch.float32)

                out_fp32 = ggml_dequantize_torch(
                    packed, int(weight_type), rows, cols, torch.float32
                )
                self.assertEqual(out_fp32.dtype, torch.float32)
                self.assertEqual(tuple(out_fp32.shape), (rows, cols))
                torch.testing.assert_close(out_fp32, ref_fp32, rtol=1e-5, atol=1e-5)

                for low_dtype in (torch.float16, torch.bfloat16):
                    out_low = ggml_dequantize_torch(
                        packed, int(weight_type), rows, cols, low_dtype
                    )
                    self.assertEqual(out_low.dtype, low_dtype)
                    torch.testing.assert_close(
                        out_low, ref_fp32.to(low_dtype), rtol=0, atol=0
                    )

    def test_dequantize_gguf_weight_and_linear_match_reference(self):
        rows, cols = 8, 256
        gen = torch.Generator().manual_seed(42)
        x = torch.randn(3, cols, generator=gen, dtype=torch.bfloat16)
        bias = torch.randn(rows, generator=gen, dtype=torch.bfloat16)

        for weight_type in (
            WeightType.Q8_0,
            WeightType.Q4_K,
            WeightType.Q6_K,
            WeightType.Q4_0,
        ):
            with self.subTest(weight_type=weight_type.name):
                packed = _make_packed_blocks(weight_type, rows, cols, seed=99)
                ref_w = torch.from_numpy(
                    gguf.dequantize(packed.numpy(), weight_type).copy()
                ).to(torch.bfloat16)
                ref_out = F.linear(x, ref_w, bias)

                dequant_w = dequantize_gguf_weight(
                    packed, int(weight_type), torch.bfloat16
                )
                torch.testing.assert_close(dequant_w, ref_w, rtol=0, atol=0)

                meta = GGUFTensorMeta(
                    ggml_type=int(weight_type),
                    logical_shape=(rows, cols),
                    stored_shape=tuple(packed.shape),
                    stored_dtype=torch.uint8,
                    param_name="w.qweight",
                )
                config = GGUFConfig("/dev/null", {"w.weight": meta})
                layer = ReplicatedLinear(
                    cols,
                    rows,
                    bias=True,
                    params_dtype=torch.bfloat16,
                    quant_config=config,
                    prefix="w",
                )
                self.assertIsInstance(layer.quant_method, GGUFLinearMethod)
                layer.qweight.data.copy_(packed)
                layer.bias.data.copy_(bias)
                actual_out, _ = layer(x)
                torch.testing.assert_close(actual_out, ref_out, rtol=1e-2, atol=1e-2)

    def test_q8_0_and_q4_0_quantize_round_trip(self):
        rows, cols = 4, 256
        gen = torch.Generator().manual_seed(7)
        weight_fp32 = torch.randn(rows, cols, generator=gen, dtype=torch.float32)
        x = torch.randn(2, cols, generator=gen, dtype=torch.float32)

        for weight_type in (WeightType.Q8_0, WeightType.Q4_0):
            with self.subTest(weight_type=weight_type.name):
                packed_np = gguf.quants.quantize(weight_fp32.numpy(), weight_type)
                packed = torch.from_numpy(packed_np.copy())
                ref_w = torch.from_numpy(
                    gguf.dequantize(packed_np, weight_type).copy()
                ).to(torch.float32)

                actual_w = dequantize_gguf_weight(
                    packed, int(weight_type), torch.float32
                )
                torch.testing.assert_close(actual_w, ref_w, rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(
                    F.linear(x, actual_w), F.linear(x, ref_w), rtol=1e-5, atol=1e-5
                )

    def test_apply_gguf_embedding_uses_torch_dequantizer(self):
        vocab_size, hidden_size = 5, 256
        packed = _make_packed_blocks(WeightType.Q4_K, vocab_size, hidden_size, seed=11)
        ref_table = torch.from_numpy(
            gguf.dequantize(packed.numpy(), WeightType.Q4_K).copy()
        ).to(torch.bfloat16)
        tokens = torch.tensor([[3, 1], [0, 4]], dtype=torch.long)

        actual = apply_gguf_embedding(
            tokens,
            packed,
            int(WeightType.Q4_K),
            hidden_size,
            dtype=torch.bfloat16,
        )
        expected = F.embedding(tokens, ref_table)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        meta = GGUFTensorMeta(
            ggml_type=int(WeightType.Q4_K),
            logical_shape=(vocab_size, hidden_size),
            stored_shape=tuple(packed.shape),
            stored_dtype=torch.uint8,
            param_name="tok.qweight",
        )
        method = GGUFEmbeddingMethod(meta, "tok")
        method.params_dtype = torch.bfloat16
        layer = torch.nn.Module()
        layer.register_parameter(
            "qweight", torch.nn.Parameter(packed, requires_grad=False)
        )
        torch.testing.assert_close(method.embedding(layer, tokens), expected)

    def test_unsupported_types_and_bad_inputs_raise_explicitly(self):
        packed = torch.zeros((4, 32), dtype=torch.uint8)
        for bad_type in (
            WeightType.IQ4_NL,
            WeightType.IQ2_XXS,
            WeightType.TQ1_0,
            WeightType.Q8_1,
            999,
        ):
            with (
                self.subTest(bad_type=bad_type),
                self.assertRaises(NotImplementedError),
            ):
                ggml_dequantize_torch(packed, int(bad_type), 4, 32, torch.float32)

        with self.assertRaises(TypeError):
            ggml_dequantize_torch(
                torch.zeros((4, 34), dtype=torch.int8),
                int(WeightType.Q8_0),
                4,
                32,
                torch.float32,
            )

        with self.assertRaises(ValueError):
            ggml_dequantize_torch(
                torch.zeros((4, 34), dtype=torch.uint8),
                int(WeightType.Q8_0),
                4,
                30,
                torch.float32,
            )

        with self.assertRaises(ValueError):
            ggml_dequantize_torch(
                torch.zeros((4, 30), dtype=torch.uint8),
                int(WeightType.Q8_0),
                4,
                32,
                torch.float32,
            )


if __name__ == "__main__":
    unittest.main()
