import asyncio
import json
import unittest
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.benchmark import serving, stall_trace
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestStallTrace(unittest.IsolatedAsyncioTestCase):
    async def test_native_raw_gap_preserves_chunk_time_without_changing_itl(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                clock = [10.0]

                async def chunks():
                    for at, count in ((11.0, 1), (15.0, 5)):
                        clock[0] = at
                        yield json.dumps(
                            {
                                "text": "private-text",
                                "meta_info": {
                                    "completion_tokens": count,
                                    "prompt_tokens": 4,
                                },
                            }
                        ).encode()

                response = MagicMock(status=200)
                response.content = chunks()
                session = MagicMock()
                session.__aenter__.return_value = session
                session.post.return_value.__aenter__.return_value = response
                serving.set_global_args(
                    Namespace(
                        backend="sglang",
                        trace_stalls=enabled,
                        temperature=0,
                        disable_ignore_eos=False,
                        disable_stream=False,
                        return_logprob=False,
                        return_routed_experts=False,
                        logprob_start_len=-1,
                        top_p=1,
                        top_logprobs_num=0,
                        token_ids_logprob=None,
                        cache_report=False,
                    )
                )
                request = SimpleNamespace(
                    api_url="http://localhost/generate",
                    prompt="private-prompt",
                    output_len=5,
                    prompt_len=4,
                    lora_name=None,
                    extra_request_body={},
                    image_data=None,
                    routing_key=None,
                )
                with (
                    patch.object(
                        serving, "_create_bench_client_session", return_value=session
                    ),
                    patch.object(serving, "get_request_headers", return_value={}),
                    patch.object(
                        serving.time, "perf_counter", side_effect=lambda: clock[0]
                    ),
                ):
                    output = await serving.async_request_sglang_generate(request)
                self.assertTrue(output.success, output.error)
                self.assertEqual(output.itl, [1.0] * 4)
                self.assertEqual(output.first_token_time, 11.0)
                self.assertEqual(
                    output.stream_gaps, [(11.0, 15.0, 4)] if enabled else []
                )
                trace = stall_trace.stall_trace_details([output], 10.0, 16.0, [])
                self.assertNotIn("private-", json.dumps(trace))

    async def test_loop_lag_excludes_intended_sleep(self):
        calls = 0

        async def sleep(_):
            nonlocal calls
            calls += 1
            if calls > 1:
                raise asyncio.CancelledError

        lags = []
        with (
            patch.object(stall_trace.asyncio, "sleep", side_effect=sleep),
            patch.object(
                stall_trace.time, "perf_counter", side_effect=[100.0, 100.35, 100.35]
            ),
            self.assertRaises(asyncio.CancelledError),
        ):
            await stall_trace._watch_event_loop(lags)
        self.assertAlmostEqual(lags[0][0], 100.05)
        self.assertAlmostEqual(lags[0][1], 100.35)

    async def test_monitor_is_cancelled_on_success_and_request_failure(self):
        for fail in (False, True):
            stopped = []

            async def watcher(_):
                try:
                    await asyncio.Event().wait()
                finally:
                    stopped.append(True)

            async def request():
                await asyncio.sleep(0)
                if fail:
                    raise ValueError("request failed")
                return 7

            with patch.object(stall_trace, "_watch_event_loop", watcher):
                if fail:
                    with self.assertRaises(ValueError):
                        await stall_trace.gather_with_stall_trace([request()], True)
                else:
                    self.assertEqual(
                        await stall_trace.gather_with_stall_trace([request()], True),
                        ([7], []),
                    )
            self.assertEqual(stopped, [True])

    def test_trace_rejects_other_clients_and_enables_details(self):
        with self.assertRaises(ValueError):
            serving.set_global_args(Namespace(trace_stalls=True, backend="vllm"))
        args = Namespace(trace_stalls=True, backend="sglang", output_details=False)
        serving.set_global_args(args)
        self.assertTrue(args.output_details)


if __name__ == "__main__":
    unittest.main()
