"""Human- and machine-readable reports.

Every writer takes the target stream explicitly, so tests can capture output
and the CLI can send progress to stderr while results go to stdout.
"""

import csv
import json
import sys
from collections import Counter
from typing import TextIO

from httpbench.network import Attempt
from httpbench.stats import BYTES_PER_MB, PerformanceStats, calculate_stats

# Layout constants. 78 characters fit comfortably in an 80-column terminal.
SUMMARY_WIDTH = 78
MAX_FAILURE_REASONS = 5

CSV_COLUMNS = (
    "number",
    "ok",
    "status",
    "elapsed_seconds",
    "bytes_received",
    "redirects",
    "reused_connection",
    "error",
    "dns_seconds",
    "connect_seconds",
    "tls_seconds",
    "ttfb_seconds",
    "body_read_seconds",
)


def _ms(seconds: float) -> str:
    return f"{seconds * 1000:,.2f} ms"


def _ms_or_dash(seconds: float | None) -> str:
    return "-" if seconds is None else _ms(seconds)


def _label_width(total: int) -> int:
    """Width of the widest ``[n/total]`` progress label."""
    return max(8, len(f"[{total}/{total}]"))


def _attempt_error(attempt: Attempt) -> str:
    """Return a printable failure reason for a failed attempt."""
    if attempt.error:
        return attempt.error
    if attempt.status is not None:
        return f"HTTP {attempt.status}"
    return "unknown error"


def print_run_header(
    url: str,
    request_count: int,
    timeout_seconds: float,
    warmup_count: int = 0,
    keep_alive: bool = False,
    delay_seconds: float = 0.0,
    cache_bust: bool = False,
    stream: TextIO = sys.stdout,
) -> None:
    """Print target and run configuration before the attempts."""
    print("Sequential HTTP download benchmark", file=stream)
    print(f"URL:      {url}", file=stream)
    print(f"Requests: {request_count}", file=stream)
    print(f"Timeout:  {timeout_seconds:g} s per request (total budget)", file=stream)
    if warmup_count:
        print(f"Warmup:   {warmup_count} request(s), not measured", file=stream)
    if keep_alive:
        print("Keep-alive: reusing the connection between requests", file=stream)
    if delay_seconds > 0:
        print(f"Delay:    {delay_seconds:g} s between requests", file=stream)
    if cache_bust:
        print("Cache busting: on", file=stream)
    print(file=stream)


def print_progress_header(request_count: int, stream: TextIO = sys.stdout) -> None:
    """Print the table header for per-request progress rows."""
    label_width = _label_width(request_count)
    header = (
        f"{'Request':<{label_width}}  {'Status':>6}  "
        f"{'Time, ms':>12}  {'Received, B':>16}  Result"
    )
    print(header, file=stream)
    print("-" * len(header), file=stream)


def print_attempt(
    attempt: Attempt, total_requests: int, stream: TextIO = sys.stdout
) -> None:
    """Print one measured progress row immediately after the attempt."""
    label_width = _label_width(total_requests)
    label = f"[{attempt.number}/{total_requests}]"
    status = f"{attempt.status:>3}" if attempt.status is not None else "---"
    outcome = "OK" if attempt.ok else f"ERROR: {attempt.error}"
    print(
        f"{label:<{label_width}}  {status:>6}  "
        f"{attempt.elapsed_seconds * 1000:>12,.2f}  "
        f"{attempt.bytes_received:>16,}  {outcome}",
        file=stream,
        flush=True,
    )


def print_warmup_attempt(
    attempt: Attempt, warmup_count: int, stream: TextIO = sys.stdout
) -> None:
    """Print one warmup progress row; warmup attempts are not measured."""
    label = f"[warmup {attempt.number}/{warmup_count}]"
    status = f"{attempt.status:>3}" if attempt.status is not None else "---"
    outcome = "OK" if attempt.ok else f"ERROR: {attempt.error}"
    print(
        f"{label:<24}  {status:>6}  "
        f"{attempt.elapsed_seconds * 1000:>12,.2f}  "
        f"{attempt.bytes_received:>16,}  {outcome}",
        file=stream,
        flush=True,
    )


def print_summary(
    attempts: list[Attempt],
    requested_count: int,
    interrupted: bool = False,
    stream: TextIO = sys.stdout,
) -> None:
    """Print aggregate statistics and grouped failure reasons."""
    successful = [attempt for attempt in attempts if attempt.ok]
    failed = [attempt for attempt in attempts if not attempt.ok]
    success_rate = len(successful) * 100.0 / requested_count if requested_count else 0.0

    print(file=stream)
    print("=" * SUMMARY_WIDTH, file=stream)
    if interrupted:
        print(
            f"SUMMARY (interrupted after {len(attempts)} of {requested_count})",
            file=stream,
        )
    else:
        print("SUMMARY", file=stream)
    print("=" * SUMMARY_WIDTH, file=stream)
    print(
        f"{'Successful requests:':<28}"
        f"{len(successful)} / {requested_count} ({success_rate:.1f}%)",
        file=stream,
    )
    print(f"{'Failed requests:':<28}{len(failed)}", file=stream)

    if successful:
        stats = calculate_stats(successful)
        print(file=stream)
        print("Request timing (successful requests only):", file=stream)
        print(f"{'  Average:':<28}{_ms(stats.average_seconds)}", file=stream)
        print(f"{'  Median:':<28}{_ms(stats.median_seconds)}", file=stream)
        print(
            f"{'  Std dev:':<28}{_ms(stats.std_dev_seconds)} "
            "(sample standard deviation)",
            file=stream,
        )
        print(
            f"{'  Min / max:':<28}{_ms(stats.minimum_seconds)} / "
            f"{_ms(stats.maximum_seconds)}",
            file=stream,
        )
        print(
            f"{'  p90 / p95:':<28}{_ms(stats.p90_seconds)} / {_ms(stats.p95_seconds)}",
            file=stream,
        )

        print(file=stream)
        print("Phases (average over successful requests):", file=stream)
        print(f"{'  DNS lookup:':<28}{_ms_or_dash(stats.dns_seconds)}", file=stream)
        print(
            f"{'  TCP connect:':<28}{_ms_or_dash(stats.connect_seconds)}", file=stream
        )
        print(f"{'  TLS handshake:':<28}{_ms_or_dash(stats.tls_seconds)}", file=stream)
        print(
            f"{'  TTFB (headers):':<28}{_ms_or_dash(stats.ttfb_seconds)}", file=stream
        )
        print(
            f"{'  Body read:':<28}{_ms(stats.body_average_seconds)} "
            "(first to last body byte)",
            file=stream,
        )
        print(
            '  Reused connections skip DNS/TCP/TLS and show as "-".',
            file=stream,
        )

        print(file=stream)
        print("Downloaded body data:", file=stream)
        print(
            f"{'  Total:':<28}{stats.total_bytes:,} bytes "
            f"({stats.total_bytes / BYTES_PER_MB:,.2f} MB)",
            file=stream,
        )

        print(file=stream)
        print("Body transfer speed (body bytes / body transfer time):", file=stream)
        print(
            "  Formula:             MB/s = body bytes / "
            "(last body byte - first body byte) / 1,000,000",
            file=stream,
        )
        if stats.body_seconds > 0:
            print(
                f"  Calculation:         {stats.total_bytes:,} B / "
                f"{stats.body_seconds:.6f} s = "
                f"{stats.body_speed_bytes_s:,.3f} B/s",
                file=stream,
            )
            print(
                f"  Speed:               {stats.body_speed_mb_s:,.3f} MB/s = "
                f"{stats.body_speed_mbit_s:,.3f} Mbit/s",
                file=stream,
            )
        else:
            print(
                "  Unavailable: no body transfer time was measured.",
                file=stream,
            )

        print(file=stream)
        print("Aggregate request speed (bytes / total request time):", file=stream)
        print(
            f"  Formula:             {stats.total_bytes:,} B / "
            f"{stats.total_seconds:.6f} s",
            file=stream,
        )
        print(
            f"  Speed:               {stats.speed_mb_s:,.3f} MB/s = "
            f"{stats.speed_mbit_s:,.3f} Mbit/s "
            "(includes DNS, TCP, TLS and headers)",
            file=stream,
        )

    if failed:
        reasons = Counter(_attempt_error(attempt) for attempt in failed)
        print(file=stream)
        print("Failure reasons:", file=stream)
        for reason, count in reasons.most_common(MAX_FAILURE_REASONS):
            print(f"  {count:>2}x  {reason}", file=stream)
        hidden_reasons = len(reasons) - MAX_FAILURE_REASONS
        if hidden_reasons > 0:
            print(f"  ... and {hidden_reasons} more distinct error(s)", file=stream)

    if not successful:
        print(file=stream)
        print(
            "No successful requests; timing and speed statistics are unavailable.",
            file=stream,
        )
    elif failed:
        print(file=stream)
        print(
            "Note: failed attempts are excluded from timing, bytes and speed.",
            file=stream,
        )

    print("=" * SUMMARY_WIDTH, file=stream)


def _attempt_to_dict(attempt: Attempt) -> dict[str, object]:
    return {
        "number": attempt.number,
        "ok": attempt.ok,
        "status": attempt.status,
        "elapsed_seconds": attempt.elapsed_seconds,
        "bytes_received": attempt.bytes_received,
        "redirects": attempt.redirects,
        "reused_connection": attempt.reused,
        "error": attempt.error,
        "phases_seconds": {
            "dns": attempt.phases.dns_seconds,
            "connect": attempt.phases.connect_seconds,
            "tls": attempt.phases.tls_seconds,
            "ttfb": attempt.phases.ttfb_seconds,
            "body_read": attempt.phases.body_seconds,
        },
    }


def _stats_to_dict(stats: PerformanceStats) -> dict[str, object]:
    return {
        "count": stats.count,
        "total_bytes": stats.total_bytes,
        "total_seconds": stats.total_seconds,
        "average_seconds": stats.average_seconds,
        "median_seconds": stats.median_seconds,
        "std_dev_seconds": stats.std_dev_seconds,
        "minimum_seconds": stats.minimum_seconds,
        "maximum_seconds": stats.maximum_seconds,
        "p90_seconds": stats.p90_seconds,
        "p95_seconds": stats.p95_seconds,
        "body_seconds": stats.body_seconds,
        "speed_bytes_s": stats.speed_bytes_s,
        "speed_mb_s": stats.speed_mb_s,
        "speed_mbit_s": stats.speed_mbit_s,
        "body_speed_bytes_s": stats.body_speed_bytes_s,
        "body_speed_mb_s": stats.body_speed_mb_s,
        "body_speed_mbit_s": stats.body_speed_mbit_s,
        "phases_seconds": {
            "dns": stats.dns_seconds,
            "connect": stats.connect_seconds,
            "tls": stats.tls_seconds,
            "ttfb": stats.ttfb_seconds,
        },
    }


def write_json_report(
    url: str,
    attempts: list[Attempt],
    requested_count: int,
    interrupted: bool,
    stream: TextIO = sys.stdout,
) -> None:
    """Write the whole run as one JSON document (machine-readable)."""
    successful = [attempt for attempt in attempts if attempt.ok]
    payload: dict[str, object] = {
        "url": url,
        "requested_requests": requested_count,
        "completed_requests": len(attempts),
        "interrupted": interrupted,
        "attempts": [_attempt_to_dict(attempt) for attempt in attempts],
        "summary": _stats_to_dict(calculate_stats(successful)) if successful else None,
    }
    json.dump(payload, stream, indent=2)
    stream.write("\n")


def _csv_value(seconds: float | None) -> str:
    return "" if seconds is None else f"{seconds:.6f}"


def write_csv_report(attempts: list[Attempt], stream: TextIO = sys.stdout) -> None:
    """Write one CSV row per attempt (machine-readable)."""
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for attempt in attempts:
        writer.writerow(
            [
                attempt.number,
                attempt.ok,
                "" if attempt.status is None else attempt.status,
                f"{attempt.elapsed_seconds:.6f}",
                attempt.bytes_received,
                attempt.redirects,
                attempt.reused,
                attempt.error or "",
                _csv_value(attempt.phases.dns_seconds),
                _csv_value(attempt.phases.connect_seconds),
                _csv_value(attempt.phases.tls_seconds),
                _csv_value(attempt.phases.ttfb_seconds),
                _csv_value(attempt.phases.body_seconds),
            ]
        )
