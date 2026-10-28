import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import sglang.srt.mem_cache.memory_pool as memory_pool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

OPTION = "SGLANG_NPU_MAMBA_CLEAR_USE_INDEX_COPY"


def make_pool(dtype=torch.bfloat16, strided=False, empty_temporal=False):
    pool = memory_pool.MambaPool.__new__(memory_pool.MambaPool)
    shapes = [(3, 7, 3, 8), (3, 7, 2, 5), (3, 7, 2, 3, 4)]
    backing, states = [], []
    for shape in shapes:
        if strided:
            shape = (*shape[:-1], shape[-1] * 2)
        tensor = (torch.arange(torch.tensor(shape).prod()).reshape(shape) % 53 + 1).to(
            dtype
        )
        backing.append(tensor)
        states.append(tensor[..., ::2] if strided else tensor)
    if empty_temporal:
        states[-1] = torch.empty(3, 7, 0, 0, 0, dtype=dtype)
        backing[-1] = states[-1]
    pool.mamba_cache = SimpleNamespace(conv=states[:2], temporal=states[2])
    pool._slot_siblings = [Mock()]
    return pool, states, backing


class TestNPUMambaClear(unittest.TestCase):
    def run_clear(self, pool, indices, enabled=True, is_npu=True):
        copies, fills = [], []
        original_copy = torch.Tensor.index_copy_
        original_fill = torch.Tensor.index_fill_

        def index_copy(tensor, dim, index, source):
            copies.append((tensor, dim, index.dtype, source.shape, source.dtype))
            return original_copy(tensor, dim, index, source)

        def index_fill(tensor, dim, index, value):
            fills.append(tensor)
            return original_fill(tensor, dim, index, value)

        with (
            patch.dict(os.environ, {OPTION: "1" if enabled else "0"}),
            patch.object(memory_pool, "_is_npu", is_npu),
            patch.object(torch.Tensor, "index_copy_", index_copy),
            patch.object(torch.Tensor, "index_fill_", index_fill),
        ):
            pool.clear_slots(indices)
        return copies, fills

    def test_npu_bf16_clears_only_selected_slots_across_all_layers(self):
        for slots in ([6, 0, 3], [4, 1, 4], list(reversed(range(7)))):
            with self.subTest(slots=slots):
                pool, states, _ = make_pool()
                expected = [state.clone() for state in states]
                for state in expected:
                    for slot in slots:
                        state[:, slot] = 0
                indices = torch.tensor(slots, dtype=torch.int32)
                copies, fills = self.run_clear(pool, indices)
                self.assertEqual(len(copies), 3)
                self.assertEqual(fills, [])
                for actual, reference, call in zip(states, expected, copies):
                    self.assertTrue(torch.equal(actual, reference))
                    self.assertIs(call[0], actual)
                    self.assertEqual(call[1:3], (1, torch.int64))
                    self.assertEqual(tuple(call[3]), (3, len(slots), *actual.shape[2:]))
                    self.assertEqual(call[4], actual.dtype)
                pool._slot_siblings[0].reset_slots.assert_called_once_with(indices)

    def test_empty_indices_and_empty_temporal_do_not_allocate_zero_sources(self):
        pool, states, _ = make_pool(empty_temporal=True)
        original = [state.clone() for state in states]
        copies, _ = self.run_clear(pool, torch.empty(0, dtype=torch.int32))
        self.assertEqual(copies, [])
        for state, reference in zip(states, original):
            self.assertTrue(torch.equal(state, reference))
        copies, _ = self.run_clear(pool, torch.tensor([6], dtype=torch.int64))
        self.assertEqual(len(copies), 2)

    def test_fallback_preserves_other_devices_dtypes_and_strided_storage(self):
        variants = [
            (False, True, torch.bfloat16, False),
            (True, False, torch.bfloat16, False),
            (True, True, torch.float32, False),
            (True, True, torch.float16, False),
            (True, True, torch.bfloat16, True),
        ]
        for enabled, is_npu, dtype, strided in variants:
            with self.subTest(
                enabled=enabled, is_npu=is_npu, dtype=dtype, strided=strided
            ):
                pool, _, backing = make_pool(dtype, strided)
                expected = [tensor.clone() for tensor in backing]
                for tensor in expected:
                    view = tensor[..., ::2] if strided else tensor
                    view[:, 2] = 0
                copies, fills = self.run_clear(pool, torch.tensor([2]), enabled, is_npu)
                self.assertEqual(copies, [])
                self.assertEqual(len(fills), 3)
                for actual, reference in zip(backing, expected):
                    self.assertTrue(torch.equal(actual, reference))

    def test_repeated_clear_uses_updated_indices_and_preserves_other_live_state(self):
        pool, states, _ = make_pool()
        original = [state.clone() for state in states]
        indices = torch.tensor([0, 3], dtype=torch.int32)
        self.run_clear(pool, indices)
        for state, reference in zip(states, original):
            state.copy_(reference)
        indices.copy_(torch.tensor([6, 2], dtype=torch.int32))
        copies, _ = self.run_clear(pool, indices)
        self.assertEqual(len(copies), 3)
        for state, reference in zip(states, original):
            reference[:, 6] = 0
            reference[:, 2] = 0
            self.assertTrue(torch.equal(state, reference))

    def test_option_is_resolved_once_per_pool(self):
        pool, _, _ = make_pool()
        self.assertEqual(
            len(self.run_clear(pool, torch.tensor([1]), enabled=True)[0]), 3
        )
        self.assertEqual(
            len(self.run_clear(pool, torch.tensor([2]), enabled=False)[0]), 3
        )
        fresh, _, _ = make_pool()
        self.assertEqual(self.run_clear(fresh, torch.tensor([1]), enabled=False)[0], [])

    def test_cuda_fused_conv_dispatch_and_sibling_reset_are_preserved(self):
        pool, states, _ = make_pool()
        pool._conv_fuse_ok = True
        pool._conv_slot_desc = object()
        order = []
        pool._slot_siblings[0].reset_slots.side_effect = lambda _: order.append(
            "sibling"
        )
        fused = Mock(side_effect=lambda *_: order.append("fused"))
        module = SimpleNamespace(fused_clear_conv_slots=fused)
        indices = torch.tensor([6, 1], dtype=torch.int32)
        with (
            patch.dict(
                "sys.modules", {"sglang.srt.mem_cache.mamba_slot_fused": module}
            ),
            patch.dict(os.environ, {"SGLANG_DISABLE_FUSED_MAMBA_SLOT_OPS": "0"}),
        ):
            copies, fills = self.run_clear(pool, indices, is_npu=False)
        self.assertEqual(copies, [])
        self.assertEqual(len(fills), 1)
        self.assertIs(fills[0], states[-1])
        self.assertEqual(order, ["sibling", "fused"])
        self.assertIs(fused.call_args.args[0], pool._conv_slot_desc)
        self.assertTrue(torch.equal(fused.call_args.args[1], indices.long()))


if __name__ == "__main__":
    unittest.main()
