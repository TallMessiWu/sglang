"""Opt-in host timing and allocator counters around NPU state clears.

No synchronize(), device events, tensor reads or allocator resets. Host spans
include enqueue/wait time, NOT asynchronous kernel completion time. Logging and
memory_stats add overhead, so these runs are diagnostic only.
"""

import json
import logging
import math
import os
import socket
import time
from contextlib import contextmanager

import torch

logger = logging.getLogger(__name__)
_HOST = socket.gethostname()
_COUNTERS = (
    "num_alloc_retries",
    "num_ooms",
    "allocated_bytes.all.current",
    "reserved_bytes.all.current",
)


def _stats(device):
    try:
        stats = torch.npu.memory_stats(device)
        return {key: stats.get(key) for key in _COUNTERS}
    except Exception as exc:
        # Diagnostic availability must not alter the inference path.
        return {"error_type": type(exc).__name__}


@contextmanager
def trace_clear(state, indices, use_index_copy):
    entered = time.perf_counter()
    before = _stats(state.device)
    record = {
        "version": 1,
        "clock": "perf_counter",
        "host": _HOST,
        "pid": os.getpid(),
        "device": str(state.device),
        "method": "index_copy" if use_index_copy else "index_fill",
        "shape": list(state.shape),
        "slots": indices.numel(),
        "zero_bytes": (
            state.shape[0]
            * indices.numel()
            * math.prod(state.shape[2:])
            * state.element_size()
            if use_index_copy
            else 0
        ),
        "entered": entered,
        "before": before,
        "started": time.perf_counter(),
    }
    try:
        yield record
    finally:
        record["ended"] = time.perf_counter()
        record["after"] = _stats(state.device)
        record["exited"] = time.perf_counter()
        logger.info(
            "[NPU_MAMBA_CLEAR_TRACE] %s", json.dumps(record, separators=(",", ":"))
        )
