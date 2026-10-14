"""CPU file-format regressions for ModelSlim Wan repacking."""

import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

from sglang.multimodal_gen.tools.wan_repack import convert_transformer, repack
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestWanRepack(unittest.TestCase):
    def write_quant(self, directory):
        directory.mkdir(parents=True)
        tensors = {
            "blocks.0.self_attn.q.weight": torch.arange(128, dtype=torch.uint8)
            .reshape(2, 64)
            .view(torch.float8_e4m3fn),
            "blocks.0.self_attn.q.weight_scale": torch.tensor(
                [[127, 128], [126, 127]], dtype=torch.uint8
            ),
            "blocks.0.self_attn.q_rot": torch.eye(2, dtype=torch.bfloat16),
            "blocks.0.self_attn.k_rot": -torch.eye(2, dtype=torch.bfloat16),
            "blocks.0.ffn.0.linear.weight": torch.ones(2, 32, dtype=torch.uint8),
            "blocks.0.ffn.0.div.mul_scale": torch.ones(2, dtype=torch.float32),
        }
        description = {
            "blocks.0.self_attn.q.weight": "W8A8_MXFP8",
            "blocks.0.self_attn.q_rot": "FLOAT",
            "blocks.0.self_attn.k_rot": "FLOAT",
            "blocks.0.self_attn.quant_type": "FP8_DYNAMIC",
            "blocks.0.cross_attn.quant_type": "FP8_DYNAMIC",
            "blocks.0.ffn.0.linear.weight": "W4A4_MXFP4",
            "producer": "modelslim",
        }
        save_file(tensors, directory / "quant_model_weight_w8a8_mxfp8.safetensors")
        (directory / "quant_model_description_w8a8_mxfp8.json").write_text(
            json.dumps(description)
        )
        return tensors, description

    def test_rotation_names_and_tensor_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tensors, _ = self.write_quant(root / "quant")
            convert_transformer("Wan2.2-TI2V-5B", root / "quant", root / "out")
            converted = load_file(root / "out/diffusion_pytorch_model.safetensors")
            names = {
                "blocks.0.self_attn.q.weight": "blocks.0.attn1.to_q.weight",
                "blocks.0.self_attn.q.weight_scale": "blocks.0.attn1.to_q.weight_scale",
                "blocks.0.self_attn.q_rot": "blocks.0.attn1.q_rot",
                "blocks.0.self_attn.k_rot": "blocks.0.attn1.k_rot",
                "blocks.0.ffn.0.linear.weight": "blocks.0.ffn.net.0.proj.weight",
                "blocks.0.ffn.0.div.mul_scale": "blocks.0.ffn.net.0.proj.mul_scale",
            }
            self.assertEqual(set(converted), set(names.values()))
            for old, new in names.items():
                self.assertEqual(converted[new].dtype, tensors[old].dtype)
                self.assertEqual(converted[new].shape, tensors[old].shape)
                self.assertTrue(
                    torch.equal(
                        converted[new].view(torch.uint8), tensors[old].view(torch.uint8)
                    )
                )
            description = json.loads(
                (root / "out/quant_model_description.json").read_text()
            )
            # These exact names enable offline Q/K rotation in WanTransformerBlock.
            self.assertEqual(description["blocks.0.attn1.q_rot"], "FLOAT")
            self.assertEqual(description["blocks.0.attn1.k_rot"], "FLOAT")

    def test_description_only_keys_are_renamed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, original = self.write_quant(root / "quant")
            convert_transformer("Wan2.2-TI2V-5B", root / "quant", root / "out")
            converted = json.loads(
                (root / "out/quant_model_description.json").read_text()
            )
            self.assertEqual(converted["blocks.0.attn1.quant_type"], "FP8_DYNAMIC")
            self.assertEqual(converted["blocks.0.attn2.quant_type"], "FP8_DYNAMIC")
            self.assertEqual(converted["producer"], "modelslim")
            self.assertEqual(len(converted), len(original))
            self.assertFalse(
                any("self_attn" in key or "cross_attn" in key for key in converted)
            )
            self.assertEqual(
                json.loads(
                    (root / "quant/quant_model_description_w8a8_mxfp8.json").read_text()
                ),
                original,
            )

    def test_repack_copies_components_and_transformer_configs(self):
        for model_type, components in (
            ("Wan2.2-TI2V-5B", [("", "transformer")]),
            (
                "Wan2.2-I2V-A14B",
                [
                    ("high_noise_model", "transformer"),
                    ("low_noise_model", "transformer_2"),
                ],
            ),
        ):
            with (
                self.subTest(model_type=model_type),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                original = root / "original"
                original.mkdir()
                (original / "model_index.json").write_text(
                    '{"_class_name": "WanPipeline"}'
                )
                (original / "vae").mkdir()
                (original / "vae/weights.bin").write_bytes(b"unchanged component")
                for quant_dir, transformer_dir in components:
                    self.write_quant(root / "quant" / quant_dir)
                    (original / transformer_dir).mkdir()
                    (original / transformer_dir / "config.json").write_text(
                        '{"num_layers": 30}'
                    )
                    (original / transformer_dir / "old.safetensors").write_bytes(
                        b"excluded"
                    )
                repack(model_type, original, root / "quant", root / "out")
                self.assertEqual(
                    (root / "out/vae/weights.bin").read_bytes(), b"unchanged component"
                )
                for _, transformer_dir in components:
                    self.assertEqual(
                        (root / "out" / transformer_dir / "config.json").read_text(),
                        '{"num_layers": 30}',
                    )
                    self.assertFalse(
                        (root / "out" / transformer_dir / "old.safetensors").exists()
                    )
                    self.assertTrue(
                        (
                            root
                            / "out"
                            / transformer_dir
                            / "diffusion_pytorch_model.safetensors"
                        ).is_file()
                    )

    def test_incomplete_template_fails_before_output_is_created(self):
        for missing in ("model_index.json", "transformer/config.json"):
            with (
                self.subTest(missing=missing),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                original = root / "original"
                (original / "transformer").mkdir(parents=True)
                for name in ("model_index.json", "transformer/config.json"):
                    if name != missing:
                        (original / name).write_text("{}")
                self.write_quant(root / "quant")
                with self.assertRaisesRegex(FileNotFoundError, missing):
                    repack("Wan2.2-TI2V-5B", original, root / "quant", root / "out")
                self.assertFalse((root / "out").exists())

    def write_ambiguous_quant(self, directory):
        tensors, description = self.write_quant(directory)
        alternative = {
            key: "FLOAT" if key.endswith(".weight") else value
            for key, value in description.items()
        }
        (directory / "quant_model_description_w8a8_fp8_only.json").write_text(
            json.dumps(alternative)
        )
        return tensors, description

    def test_multiple_descriptions_require_selection_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_ambiguous_quant(root / "quant")
            with self.assertRaisesRegex(ValueError, "--quant-description-file"):
                convert_transformer("Wan2.2-TI2V-5B", root / "quant", root / "out")
            self.assertFalse((root / "out").exists())

    def test_cli_selects_description_for_each_expert(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "original"
            original.mkdir()
            (original / "model_index.json").write_text("{}")
            for index, (expert, component) in enumerate(
                (
                    ("high_noise_model", "transformer"),
                    ("low_noise_model", "transformer_2"),
                )
            ):
                _, description = self.write_ambiguous_quant(root / "quant" / expert)
                # Distinct metadata proves each expert uses its own selected file.
                description["expert"] = index
                (
                    root / "quant" / expert / "quant_model_description_w8a8_mxfp8.json"
                ).write_text(json.dumps(description))
                (original / component).mkdir()
                (original / component / "config.json").write_text("{}")
            tool = sys.modules[convert_transformer.__module__].__file__
            with patch.object(
                sys,
                "argv",
                [
                    tool,
                    "--model-type",
                    "Wan2.2-I2V-A14B",
                    "--original-model-path",
                    str(original),
                    "--quant-path",
                    str(root / "quant"),
                    "--output-path",
                    str(root / "out"),
                    "--quant-description-file",
                    "quant_model_description_w8a8_mxfp8.json",
                ],
            ):
                runpy.run_path(tool, run_name="__main__")
            for index, component in enumerate(("transformer", "transformer_2")):
                description = json.loads(
                    (
                        root / "out" / component / "quant_model_description.json"
                    ).read_text()
                )
                self.assertEqual(
                    description["blocks.0.attn1.to_q.weight"], "W8A8_MXFP8"
                )
                self.assertEqual(description["expert"], index)

    def test_description_selection_errors_before_copying_template(self):
        for selection in (None, "missing.json", "../outside.json"):
            with (
                self.subTest(selection=selection),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                original = root / "original"
                (original / "transformer").mkdir(parents=True)
                (original / "model_index.json").write_text("{}")
                (original / "transformer/config.json").write_text("{}")
                self.write_ambiguous_quant(root / "quant")
                with self.assertRaises((ValueError, FileNotFoundError)):
                    repack(
                        "Wan2.2-TI2V-5B",
                        original,
                        root / "quant",
                        root / "out",
                        quant_description_file=selection,
                    )
                self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()
