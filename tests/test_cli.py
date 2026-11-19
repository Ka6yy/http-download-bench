"""Unit and subprocess tests for httpbench.cli."""

import functools
import io
import json
import os
import subprocess
import sys
import time
import unittest
import urllib.parse
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from helpers import TestServer

from httpbench import cli
from httpbench.cli import (
    HttpClient,
    RunOptions,
    build_parser,
    cache_bust_url,
    http_url,
    main,
    run_benchmark,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_PY = REPO_ROOT / "main.py"


class ParserTests(unittest.TestCase):
    def test_defaults(self) -> None:
        args = build_parser().parse_args(["https://example.com/f.jpg"])

        self.assertEqual(args.requests, 10)
        self.assertEqual(args.timeout, 30.0)
        self.assertEqual(args.warmup, 0)
        self.assertFalse(args.keep_alive)
        self.assertEqual(args.delay, 0.0)
        self.assertFalse(args.cache_bust)
        self.assertFalse(args.json)
        self.assertFalse(args.csv)

    def test_overrides(self) -> None:
        args = build_parser().parse_args(
            [
                "https://example.com/f.jpg",
                "-n",
                "7",
                "--timeout",
                "12.5",
                "--warmup",
                "2",
                "--keep-alive",
                "--delay",
                "0.25",
                "--cache-bust",
                "--json",
            ]
        )

        self.assertEqual(args.url, "https://example.com/f.jpg")
        self.assertEqual(args.requests, 7)
        self.assertEqual(args.timeout, 12.5)
        self.assertEqual(args.warmup, 2)
        self.assertTrue(args.keep_alive)
        self.assertEqual(args.delay, 0.25)
        self.assertTrue(args.cache_bust)
        self.assertTrue(args.json)

    def test_json_and_csv_are_mutually_exclusive(self) -> None:
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as context:
                build_parser().parse_args(["https://example.com/", "--json", "--csv"])

        self.assertEqual(context.exception.code, 2)

    def test_rejects_out_of_range_values(self) -> None:
        bad_invocations = (
            ["https://example.com/", "-n", "0"],
            ["https://example.com/", "--warmup", "-1"],
            ["https://example.com/", "--delay", "-1"],
            ["https://example.com/", "--timeout", "0"],
            ["https://example.com/", "--requests", "abc"],
        )
        for invocation in bad_invocations:
            with self.subTest(invocation=invocation):
                with redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as context:
                        build_parser().parse_args(invocation)
                self.assertEqual(context.exception.code, 2)


class UrlValidationTests(unittest.TestCase):
    def test_accepts_valid_urls(self) -> None:
        for url in (
            "http://example.com/file.bin",
            "https://example.com:8443/file.bin?range=10",
            "HTTP://EXAMPLE.COM/file.bin",
            "http://[::1]:8080/file.bin",
            "http://user:pass@example.com/file.bin",
        ):
            with self.subTest(url=url):
                self.assertEqual(http_url(url), url)

    def test_rejects_invalid_urls(self) -> None:
        import argparse

        for url in (
            "ftp://example.com/file.bin",
            "example.com/file.bin",  # No scheme.
            "http:///file.bin",  # No host.
            "http://:8080/file.bin",  # No host.
            "http://exa mple.com/file.bin",  # Whitespace.
            "http://example.com:notaport/file.bin",
            "http://[::1/file.bin",  # Broken IPv6 literal.
        ):
            with self.subTest(url=url):
                with self.assertRaises(argparse.ArgumentTypeError):
                    http_url(url)


class CacheBustTests(unittest.TestCase):
    def test_appends_parameter(self) -> None:
        busted = cache_bust_url("http://example.com/f.bin", 1)
        parsed = urllib.parse.urlsplit(busted)
        self.assertIn("_=", parsed.query)
        self.assertEqual(parsed.path, "/f.bin")

    def test_is_unique_per_call(self) -> None:
        first = cache_bust_url("http://example.com/f.bin", 1)
        second = cache_bust_url("http://example.com/f.bin", 1)
        self.assertNotEqual(first, second)

    def test_preserves_existing_parameters(self) -> None:
        busted = cache_bust_url("http://example.com/f.bin?bytes=100&z=1", 3)
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(busted).query)
        self.assertEqual(query["bytes"], ["100"])
        self.assertEqual(query["z"], ["1"])

    def test_replaces_previous_bust_parameter(self) -> None:
        busted = cache_bust_url("http://example.com/f.bin?_=old&bytes=5", 3)
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(busted).query)
        self.assertNotIn("old", query["_"])
        self.assertEqual(query["bytes"], ["5"])


class RunBenchmarkTests(unittest.TestCase):
    """run_benchmark with captured streams (no real stderr/stdout needed)."""

    def run_with(self, url: str, **kwargs: object) -> tuple[int, str, str]:
        progress = io.StringIO()
        output = io.StringIO()
        options = RunOptions(url=url, **kwargs)  # type: ignore[arg-type]
        code = run_benchmark(options, progress_stream=progress, output_stream=output)
        return code, progress.getvalue(), output.getvalue()

    def test_success_reports_progress_to_stderr_and_summary_to_stdout(self) -> None:
        with TestServer() as server:
            code, progress, output = self.run_with(server.url_for("/ok"), requests=2)

        self.assertEqual(code, 0)
        self.assertIn("Requests:", progress)
        self.assertIn("[2/2]", progress)
        self.assertNotIn("[1/2]", output)
        self.assertIn("SUMMARY", output)
        self.assertIn("2 / 2 (100.0%)", output)

    def test_failure_returns_one(self) -> None:
        with TestServer() as server:
            code, _, output = self.run_with(server.url_for("/not-found"), requests=1)

        self.assertEqual(code, 1)
        self.assertIn("Failure reasons:", output)
        self.assertIn("HTTP 404", output)

    def test_json_output_goes_to_stdout(self) -> None:
        with TestServer() as server:
            code, progress, output = self.run_with(
                server.url_for("/ok"), requests=2, output_format="json"
            )

        self.assertEqual(code, 0)
        data = json.loads(output)
        self.assertEqual(data["completed_requests"], 2)
        self.assertTrue(all(attempt["ok"] for attempt in data["attempts"]))
        self.assertIsNotNone(data["summary"])
        self.assertNotIn("SUMMARY", output)  # Human report suppressed.
        self.assertIn("[2/2]", progress)

    def test_csv_output_goes_to_stdout(self) -> None:
        with TestServer() as server:
            code, _, output = self.run_with(
                server.url_for("/ok"), requests=2, output_format="csv"
            )

        self.assertEqual(code, 0)
        lines = output.strip().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[0].startswith("number,ok,status"))

    def test_warmup_requests_are_excluded_from_results(self) -> None:
        with TestServer() as server:
            code, progress, output = self.run_with(
                server.url_for("/ok"), requests=2, warmup=2, output_format="json"
            )

        self.assertEqual(code, 0)
        data = json.loads(output)
        self.assertEqual(data["completed_requests"], 2)
        self.assertEqual(len(data["attempts"]), 2)
        self.assertEqual(data["summary"]["count"], 2)
        self.assertIn("[warmup 2/2]", progress)
        self.assertNotIn("[warmup", output)

    def test_delay_pauses_between_requests(self) -> None:
        with TestServer() as server:
            started = time.monotonic()
            code, _, _ = self.run_with(server.url_for("/ok"), requests=3, delay=0.15)
            duration = time.monotonic() - started

        self.assertEqual(code, 0)
        # Two pauses of 0.15 s between three requests.
        self.assertGreaterEqual(duration, 0.28)

    def test_keep_alive_uses_single_connection(self) -> None:
        with TestServer() as server:
            code, _, _ = self.run_with(
                server.url_for("/ok"), requests=3, keep_alive=True
            )
            connections = server.connections_opened

        self.assertEqual(code, 0)
        self.assertEqual(connections, 1)

    def test_cache_busted_requests_all_succeed(self) -> None:
        with TestServer() as server:
            code, _, output = self.run_with(
                server.url_for("/echo-path"), requests=3, cache_bust=True
            )

        self.assertEqual(code, 0)
        self.assertIn("3 / 3 (100.0%)", output)


class _InterruptingClient(HttpClient):
    """HttpClient that raises KeyboardInterrupt after N successful requests."""

    def __init__(self, interrupt_after: int) -> None:
        super().__init__()
        self._interrupt_after = interrupt_after
        self._calls = 0

    def get(
        self,
        url: str,
        *,
        timeout_seconds: float,
        number: int = 1,
        keep_alive: bool = False,
        extra_headers: dict[str, str] | None = None,
    ):
        self._calls += 1
        if self._calls > self._interrupt_after:
            raise KeyboardInterrupt
        return super().get(
            url,
            timeout_seconds=timeout_seconds,
            number=number,
            keep_alive=keep_alive,
            extra_headers=extra_headers,
        )


class KeyboardInterruptTests(unittest.TestCase):
    def _run_interrupted(self, *extra_args: str) -> tuple[int, str]:
        with TestServer() as server:
            original = cli.HttpClient
            cli.HttpClient = functools.partial(_InterruptingClient, 1)
            try:
                captured = io.StringIO()
                with redirect_stdout(captured), redirect_stderr(io.StringIO()):
                    code = main([server.url_for("/ok"), "-n", "3", *extra_args])
            finally:
                cli.HttpClient = original
        return code, captured.getvalue()

    def test_partial_summary_and_exit_code_130(self) -> None:
        code, output = self._run_interrupted()

        self.assertEqual(code, 130)
        self.assertIn("SUMMARY (interrupted after 1 of 3)", output)
        self.assertIn("1 / 3", output)

    def test_json_partial_document_on_interrupt(self) -> None:
        code, output = self._run_interrupted("--json")

        self.assertEqual(code, 130)
        data = json.loads(output)
        self.assertTrue(data["interrupted"])
        self.assertEqual(data["completed_requests"], 1)
        self.assertTrue(data["attempts"][0]["ok"])


class SubprocessTests(unittest.TestCase):
    """Exit codes of `python main.py ...` run as a real process."""

    def run_cli(self, *args: str, timeout: float = 60.0) -> subprocess.CompletedProcess:
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        return subprocess.run(
            [sys.executable, str(MAIN_PY), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=REPO_ROOT,
            env=env,
            timeout=timeout,
        )

    def test_exit_code_zero_on_success(self) -> None:
        with TestServer() as server:
            result = self.run_cli(server.url_for("/ok"), "-n", "2", "--json")

        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["completed_requests"], 2)
        self.assertTrue(all(attempt["ok"] for attempt in data["attempts"]))

    def test_exit_code_one_on_failure(self) -> None:
        with TestServer() as server:
            result = self.run_cli(server.url_for("/not-found"), "-n", "1", "--json")

        self.assertEqual(result.returncode, 1, result.stderr)
        data = json.loads(result.stdout)
        self.assertFalse(data["attempts"][0]["ok"])

    def test_exit_code_two_on_invalid_arguments(self) -> None:
        result = self.run_cli("ftp://example.com/file.bin")

        self.assertEqual(result.returncode, 2)
        self.assertIn("URL scheme must be http:// or https://", result.stderr)

    def test_text_mode_progress_on_stderr_report_on_stdout(self) -> None:
        with TestServer() as server:
            result = self.run_cli(server.url_for("/ok"), "-n", "1")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("[1/1]", result.stderr)
        self.assertIn("SUMMARY", result.stdout)
        self.assertNotIn("[1/1]", result.stdout)

    @unittest.skipIf(os.name == "nt", "POSIX-only SIGINT delivery")
    def test_sigint_returns_130_with_partial_json(self) -> None:
        import select
        import signal

        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        with TestServer() as server:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    str(MAIN_PY),
                    server.url_for("/first-ok-then-hang"),
                    "-n",
                    "3",
                    "--json",
                    "-t",
                    "20",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                cwd=REPO_ROOT,
                env=env,
            )
            try:
                first_row = None
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    ready, _, _ = select.select([proc.stderr], [], [], 1.0)
                    if not ready:
                        continue
                    first_row = proc.stderr.readline()
                    if first_row.startswith("[1/3]"):
                        break
                self.assertIsNotNone(
                    first_row, "first progress row never appeared on stderr"
                )
                time.sleep(0.2)
                proc.send_signal(signal.SIGINT)
                stdout, _ = proc.communicate(timeout=30)
            finally:
                if proc.poll() is None:
                    proc.kill()

        self.assertEqual(proc.returncode, 130)
        data = json.loads(stdout)
        self.assertTrue(data["interrupted"])
        self.assertEqual(data["completed_requests"], 1)


if __name__ == "__main__":
    unittest.main()
