"""Opt-in client diagnostics. No device synchronization or server profiling."""

import asyncio
import socket
import time


async def _watch_event_loop(lags):
    previous = time.perf_counter()
    try:
        while True:
            await asyncio.sleep(0.05)
            now = time.perf_counter()
            # Record lateness after the intended wakeup, not the sleep itself.
            if now - previous > 0.15:
                lags.append([previous + 0.05, now])
            previous = now
    finally:
        # The last response can complete before an overdue timer resumes.
        now = time.perf_counter()
        if now - previous > 0.15:
            lags.append([previous + 0.05, now])


async def gather_with_stall_trace(tasks, enabled):
    lags = []
    watcher = asyncio.create_task(_watch_event_loop(lags)) if enabled else None
    try:
        return await asyncio.gather(*tasks), lags
    finally:
        if watcher is not None:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)


def stall_trace_details(outputs, started, ended, lags):
    return {
        "version": 1,
        "clock": "perf_counter",
        "host": socket.gethostname(),
        "started": started,
        "ended": ended,
        "request_starts": [output.start_time for output in outputs],
        "request_first_tokens": [output.first_token_time for output in outputs],
        "stream_gaps": [output.stream_gaps for output in outputs],
        "client_loop_lags": lags,
    }
