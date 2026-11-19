"""Command-line interface: argument parsing and the benchmark loop."""

import argparse
import itertools
import math
import sys
import time
import urllib.parse
from dataclasses import dataclass
from typing import TextIO

from httpbench import report
from httpbench.network import Attempt, HttpClient

DEFAULT_REQUESTS = 10
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_WARMUP = 0
DEFAULT_DELAY_SECONDS = 0.0

OUTPUT_FORMATS = ("text", "json", "csv")

# Guarantees unique cache-busting values even when the clock granularity is
# too coarse to distinguish two consecutive calls.
_bust_counter = itertools.count()


def positive_int(raw_value: str) -> int:
    """argparse type for an integer greater than or equal to 1."""
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def non_negative_int(raw_value: str) -> int:
    """argparse type for an integer greater than or equal to 0."""
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return value


def positive_seconds(raw_value: str) -> float:
    """argparse type for a finite, strictly positive number of seconds."""
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than 0")
    return value


def non_negative_seconds(raw_value: str) -> float:
    """argparse type for a finite number of seconds greater than or equal to 0."""
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError(
            "must be a finite number greater than or equal to 0"
        )
    return value


def http_url(raw_value: str) -> str:
    """argparse type that accepts only absolute HTTP(S) URLs."""
    try:
        parsed = urllib.parse.urlsplit(raw_value)
        # Touching these attributes validates IPv6 bracket and port syntax.
        _ = parsed.hostname
        _ = parsed.port
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid URL: {exc}") from exc

    if parsed.scheme.lower() not in ("http", "https"):
        raise argparse.ArgumentTypeError("URL scheme must be http:// or https://")
    if not parsed.netloc or not parsed.hostname:
        raise argparse.ArgumentTypeError("URL must include a host")
    if any(char.isspace() for char in raw_value):
        raise argparse.ArgumentTypeError("URL must not contain whitespace")
    return raw_value


def cache_bust_url(url: str, index: int) -> str:
    """Append a unique ``_`` query parameter to bypass intermediary caches."""
    parts = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
    query = [(key, value) for key, value in query if key != "_"]
    query.append(("_", f"{index}-{time.time_ns()}-{next(_bust_counter)}"))
    return urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(query)))


@dataclass(frozen=True)
class RunOptions:
    """Everything the benchmark loop needs, already validated."""

    url: str
    requests: int = DEFAULT_REQUESTS
    timeout: float = DEFAULT_TIMEOUT_SECONDS
    warmup: int = DEFAULT_WARMUP
    keep_alive: bool = False
    delay: float = DEFAULT_DELAY_SECONDS
    cache_bust: bool = False
    output_format: str = "text"


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run sequential HTTP GET requests and report latency, size and "
            "aggregate download speed."
        ),
        epilog=(
            "Units: 1 MB = 1,000,000 bytes; 1 Mbit/s = 1,000,000 bits/s. "
            "Exit status: 0 if all attempts succeed, 1 if any attempt fails, "
            "2 for invalid arguments, 130 when interrupted with Ctrl+C."
        ),
    )
    parser.add_argument(
        "url",
        type=http_url,
        help="absolute http:// or https:// URL to request",
    )
    parser.add_argument(
        "-n",
        "--requests",
        type=positive_int,
        default=DEFAULT_REQUESTS,
        metavar="COUNT",
        help=f"number of sequential requests (default: {DEFAULT_REQUESTS})",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=positive_seconds,
        default=DEFAULT_TIMEOUT_SECONDS,
        metavar="SECONDS",
        help=(
            "total timeout budget per request in seconds, including "
            "redirects (DNS lookups cannot be interrupted) "
            f"(default: {DEFAULT_TIMEOUT_SECONDS:g})"
        ),
    )
    parser.add_argument(
        "--warmup",
        type=non_negative_int,
        default=DEFAULT_WARMUP,
        metavar="COUNT",
        help=(
            "extra requests sent before measurement to warm caches and "
            "connections; excluded from statistics (default: 0)"
        ),
    )
    parser.add_argument(
        "--keep-alive",
        action="store_true",
        help=(
            "reuse the server connection across requests instead of "
            "reconnecting every time"
        ),
    )
    parser.add_argument(
        "--delay",
        type=non_negative_seconds,
        default=DEFAULT_DELAY_SECONDS,
        metavar="SECONDS",
        help="pause between measured requests (default: 0)",
    )
    parser.add_argument(
        "--cache-bust",
        action="store_true",
        help="append a unique query parameter to every request to bypass caches",
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--json",
        action="store_true",
        help="write results as JSON to stdout (progress goes to stderr)",
    )
    output_group.add_argument(
        "--csv",
        action="store_true",
        help="write one CSV row per attempt to stdout (progress goes to stderr)",
    )
    return parser


def _request_url(options: RunOptions, index: int) -> str:
    if not options.cache_bust:
        return options.url
    return cache_bust_url(options.url, index)


def run_benchmark(
    options: RunOptions,
    progress_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
) -> int:
    """Run all attempts sequentially, report results, and return an exit code.

    Progress (per-request rows) goes to stderr, the final report to stdout,
    so machine-readable output stays clean for pipes and CI. Ctrl+C prints a
    partial summary of the attempts already finished and returns 130.
    """
    progress = sys.stderr if progress_stream is None else progress_stream
    output = sys.stdout if output_stream is None else output_stream
    text_mode = options.output_format == "text"

    attempts: list[Attempt] = []
    interrupted = False
    client = HttpClient()
    try:
        # Progress (configuration, per-request rows) always goes to stderr so
        # that stdout carries exactly the report, whatever its format.
        report.print_run_header(
            options.url,
            options.requests,
            options.timeout,
            warmup_count=options.warmup,
            keep_alive=options.keep_alive,
            delay_seconds=options.delay,
            cache_bust=options.cache_bust,
            stream=progress,
        )
        report.print_progress_header(options.requests, stream=progress)

        for warmup_number in range(1, options.warmup + 1):
            warmup_attempt = client.get(
                _request_url(options, warmup_number),
                timeout_seconds=options.timeout,
                number=warmup_number,
                keep_alive=options.keep_alive,
            )
            report.print_warmup_attempt(warmup_attempt, options.warmup, stream=progress)

        for number in range(1, options.requests + 1):
            if number > 1 and options.delay > 0:
                time.sleep(options.delay)
            attempt = client.get(
                _request_url(options, number),
                timeout_seconds=options.timeout,
                number=number,
                keep_alive=options.keep_alive,
            )
            attempts.append(attempt)
            report.print_attempt(attempt, options.requests, stream=progress)

    except KeyboardInterrupt:
        interrupted = True
        if text_mode:
            print("\nInterrupted by user; partial summary below.", file=progress)
    finally:
        client.close()

    if options.output_format == "json":
        report.write_json_report(
            options.url, attempts, options.requests, interrupted, stream=output
        )
    elif options.output_format == "csv":
        report.write_csv_report(attempts, stream=output)
    else:
        report.print_summary(attempts, options.requests, interrupted, stream=output)

    if interrupted:
        return 130
    return 0 if all(attempt.ok for attempt in attempts) else 1


def configure_output() -> None:
    """Avoid encoding crashes when output exceeds the legacy console code page."""
    for stream in (sys.stdout, sys.stderr):
        # reconfigure() exists on TextIOWrapper but is not part of TextIO.
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            # Not a TextIOWrapper, for example when embedded in another app.
            continue
        try:
            reconfigure(errors="backslashreplace")
        except (OSError, ValueError):
            pass


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, handle user interruption, and run the benchmark."""
    configure_output()
    args = build_parser().parse_args(argv)
    options = RunOptions(
        url=args.url,
        requests=args.requests,
        timeout=args.timeout,
        warmup=args.warmup,
        keep_alive=args.keep_alive,
        delay=args.delay,
        cache_bust=args.cache_bust,
        output_format="json" if args.json else "csv" if args.csv else "text",
    )
    try:
        return run_benchmark(options)
    except KeyboardInterrupt:
        # Interrupted outside the attempt loop, e.g. during setup.
        print("\nInterrupted by user.", file=sys.stderr)
        return 130
