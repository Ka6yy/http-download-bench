#!/usr/bin/env python3
"""Sequential HTTP download benchmark.

Why Python:
- Python 3.9+ is widely available and needs no build step;
- urllib.request supports HTTP/HTTPS, redirects and TLS out of the box;
- a chunked read loop keeps memory usage constant for arbitrarily large files;
- the standard library covers timing, argument parsing and tests.

The body is consumed one small chunk at a time and is never saved to disk.
A failed attempt is reported, the benchmark continues, and all statistics use
successful attempts only. The process exits with status 1 if at least one
attempt failed, so partial failures can be detected by CI or scripts.
"""

import argparse
import http.client
import math
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

# Read size balances system-call overhead against constant memory use.
READ_CHUNK_BYTES = 64 * 1024

DEFAULT_REQUESTS = 10
DEFAULT_TIMEOUT_SECONDS = 30.0

# Decimal (SI) units: 1 MB = 1,000,000 bytes, 1 byte = 8 bits.
BYTES_PER_MB = 1_000_000
BITS_PER_BYTE = 8

USER_AGENT = "http-download-bench/1.1"

# Layout constants. 78 characters fit comfortably in an 80-column terminal.
SUMMARY_WIDTH = 78
MAX_ERROR_LENGTH = 180
MAX_FAILURE_REASONS = 5


class RequestDeadlineExceeded(Exception):
    """Raised when a request's total wall-clock budget is exhausted."""

    def __init__(self, received: int = 0) -> None:
        super().__init__("request deadline exceeded")
        self.received = received


class IncompleteBodyError(Exception):
    """Raised when the connection ends before the complete body is read."""

    def __init__(self, received: int, expected: Optional[int] = None) -> None:
        self.received = received
        self.expected = expected
        if expected is None:
            message = (
                "incomplete response body: connection closed after "
                f"{received:,} bytes"
            )
        else:
            message = (
                "incomplete response body: received "
                f"{received:,} of {expected:,} bytes"
            )
        super().__init__(message)


@dataclass(frozen=True)
class Attempt:
    """Result of one HTTP GET attempt."""

    number: int
    status: Optional[int]
    elapsed_seconds: float
    bytes_received: int
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        """Whether the response completed with a 2xx status."""
        return (
            self.error is None
            and self.status is not None
            and 200 <= self.status < 300
        )


@dataclass(frozen=True)
class PerformanceStats:
    """Aggregate statistics for successful attempts."""

    count: int
    total_bytes: int
    total_seconds: float
    minimum_seconds: float
    maximum_seconds: float

    @property
    def average_seconds(self) -> float:
        """Arithmetic mean request time."""
        return self.total_seconds / self.count

    @property
    def speed_bytes_s(self) -> float:
        """Aggregate throughput in bytes per second."""
        return self.total_bytes / self.total_seconds

    @property
    def speed_mb_s(self) -> float:
        """Aggregate throughput in decimal megabytes per second."""
        return self.speed_bytes_s / BYTES_PER_MB

    @property
    def speed_mbit_s(self) -> float:
        """Aggregate throughput in decimal megabits per second."""
        return self.speed_mb_s * BITS_PER_BYTE


def positive_int(raw_value: str) -> int:
    """argparse type for an integer greater than or equal to 1."""
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
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


def http_url(raw_value: str) -> str:
    """argparse type that accepts only absolute HTTP(S) URLs."""
    try:
        parsed = urllib.parse.urlsplit(raw_value)
        parsed.hostname  # Accessing hostname validates IPv6 bracket syntax.
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid URL: {exc}") from exc

    if parsed.scheme.lower() not in ("http", "https"):
        raise argparse.ArgumentTypeError("URL scheme must be http:// or https://")
    if not parsed.netloc or not parsed.hostname:
        raise argparse.ArgumentTypeError("URL must include a host")
    if any(char.isspace() for char in raw_value):
        raise argparse.ArgumentTypeError("URL must not contain whitespace")
    return raw_value


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
            "2 for invalid arguments."
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
            "total timeout budget per request in seconds "
            f"(default: {DEFAULT_TIMEOUT_SECONDS:g})"
        ),
    )
    return parser


def _is_success_status(status: Optional[int]) -> bool:
    """Return True only for HTTP 2xx statuses."""
    return status is not None and 200 <= status < 300


def _format_error(value: object) -> str:
    """Collapse an error to one line and keep progress rows readable."""
    text = " ".join(str(value).split())
    if not text:
        text = "unknown error"
    if len(text) > MAX_ERROR_LENGTH:
        text = text[: MAX_ERROR_LENGTH - 3].rstrip() + "..."
    return text


def _failure(
    number: int,
    started: float,
    status: Optional[int],
    received: int,
    error: object,
) -> Attempt:
    """Build a failed attempt, freezing the elapsed time immediately."""
    return Attempt(
        number=number,
        status=status,
        elapsed_seconds=time.perf_counter() - started,
        bytes_received=received,
        error=_format_error(error),
    )


def _describe_exception(exc: Exception, timeout_seconds: float) -> str:
    """Turn transport exceptions into short, user-facing messages."""
    if isinstance(exc, urllib.error.URLError):
        reason = exc.reason
        if isinstance(reason, (socket.timeout, TimeoutError)):
            return f"timeout after {timeout_seconds:g} s"
        return f"network error: {reason}"
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return f"timeout after {timeout_seconds:g} s"
    if isinstance(exc, http.client.RemoteDisconnected):
        return "connection closed before response headers were received"
    if isinstance(exc, http.client.HTTPException):
        return f"HTTP protocol error: {exc}"
    if isinstance(exc, ConnectionError):
        return f"connection error: {exc}"
    return f"{type(exc).__name__}: {exc}"


def _declared_content_length(response: http.client.HTTPResponse) -> Optional[int]:
    """Extract Content-Length when the body has a fixed length.

    Chunked responses are validated by http.client, so Content-Length is not
    used for them. Invalid or ambiguous headers are ignored rather than
    guessed.
    """
    transfer_encoding = response.headers.get("Transfer-Encoding", "")
    if "chunked" in transfer_encoding.lower():
        return None

    raw_length = response.headers.get("Content-Length")
    if raw_length is None:
        return None

    try:
        values = {int(part.strip()) for part in raw_length.split(",")}
    except ValueError:
        return None

    if len(values) != 1:
        return None
    value = values.pop()
    return value if value >= 0 else None


def _read_body(
    response: http.client.HTTPResponse,
    started: float,
    timeout_seconds: float,
) -> Tuple[int, float]:
    """Stream the complete body and return (byte count, finish timestamp)."""
    expected_bytes = _declared_content_length(response)

    # read1() returns available data immediately instead of waiting for a full
    # 64 KiB buffer. This makes the wall-clock deadline responsive to slowly
    # trickled responses. Fall back for unusual response objects.
    read_chunk = getattr(response, "read1", response.read)

    deadline = started + timeout_seconds
    received = 0

    while True:
        if time.perf_counter() > deadline:
            raise RequestDeadlineExceeded(received)

        try:
            chunk = read_chunk(READ_CHUNK_BYTES)
        except (socket.timeout, TimeoutError) as exc:
            raise RequestDeadlineExceeded(received) from exc
        except http.client.IncompleteRead as exc:
            raise IncompleteBodyError(
                received + len(exc.partial), expected_bytes
            ) from exc
        except (ConnectionError, http.client.HTTPException) as exc:
            raise IncompleteBodyError(received, expected_bytes) from exc

        if not chunk:
            break

        received += len(chunk)
        if time.perf_counter() > deadline:
            raise RequestDeadlineExceeded(received)

        # read1() does not flag a fixed-length body that ends early, so stop
        # once Content-Length is satisfied and validate the count below.
        if expected_bytes is not None and received >= expected_bytes:
            break

    finished_at = time.perf_counter()
    if expected_bytes is not None and received < expected_bytes:
        raise IncompleteBodyError(received, expected_bytes)
    return received, finished_at


def perform_request(url: str, timeout_seconds: float, number: int) -> Attempt:
    """Perform one GET request and receive the complete response body.

    Redirects are followed by urllib.request. The timer starts before opening
    the connection and stops when the last body byte has been received.
    """
    started = time.perf_counter()
    status: Optional[int] = None
    received = 0

    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "*/*"},
            method="GET",
        )

        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status = response.getcode()
            if not _is_success_status(status):
                return _failure(
                    number,
                    started,
                    status,
                    0,
                    f"unexpected HTTP status {status}",
                )

            received, finished_at = _read_body(response, started, timeout_seconds)

        return Attempt(
            number=number,
            status=status,
            elapsed_seconds=finished_at - started,
            bytes_received=received,
        )

    except urllib.error.HTTPError as exc:
        try:
            exc.close()
        except OSError:
            pass
        return _failure(
            number,
            started,
            exc.code,
            received,
            f"HTTP {exc.code} {exc.reason}",
        )

    except RequestDeadlineExceeded as exc:
        return _failure(
            number,
            started,
            status,
            exc.received,
            f"timeout: exceeded {timeout_seconds:g} s",
        )

    except IncompleteBodyError as exc:
        return _failure(
            number,
            started,
            status,
            exc.received,
            exc,
        )

    except Exception as exc:  # One bad response must not abort the whole run.
        return _failure(
            number,
            started,
            status,
            received,
            _describe_exception(exc, timeout_seconds),
        )


def calculate_stats(attempts: Sequence[Attempt]) -> PerformanceStats:
    """Calculate aggregate metrics from at least one successful attempt."""
    if not attempts:
        raise ValueError("at least one successful attempt is required")

    times = [attempt.elapsed_seconds for attempt in attempts]
    total_seconds = math.fsum(times)
    if total_seconds <= 0:
        raise ValueError("total elapsed time must be positive")
    return PerformanceStats(
        count=len(attempts),
        total_bytes=sum(attempt.bytes_received for attempt in attempts),
        total_seconds=total_seconds,
        minimum_seconds=min(times),
        maximum_seconds=max(times),
    )


def _attempt_error(attempt: Attempt) -> str:
    """Return a printable failure reason for a failed attempt."""
    if attempt.error:
        return attempt.error
    if attempt.status is not None:
        return f"HTTP {attempt.status}"
    return "unknown error"


def print_header(url: str, request_count: int, timeout_seconds: float) -> None:
    """Print target and run configuration before the attempts."""
    print("Sequential HTTP download benchmark")
    print(f"URL:      {url}")
    print(f"Requests: {request_count}")
    print(f"Timeout:  {timeout_seconds:g} s per request")
    print()

    label_width = max(8, len(f"[{request_count}/{request_count}]"))
    header = (
        f"{'Request':<{label_width}}  {'Status':>6}  "
        f"{'Time, ms':>12}  {'Received, B':>16}  Result"
    )
    print(header)
    print("-" * len(header))


def print_attempt(attempt: Attempt, total_requests: int) -> None:
    """Print one progress/result row immediately after the attempt finishes."""
    label_width = max(8, len(f"[{total_requests}/{total_requests}]"))
    label = f"[{attempt.number}/{total_requests}]"
    status = f"{attempt.status:>3}" if attempt.status is not None else "---"
    outcome = "OK" if attempt.ok else f"ERROR: {attempt.error}"
    print(
        f"{label:<{label_width}}  {status:>6}  "
        f"{attempt.elapsed_seconds * 1000:>12,.2f}  "
        f"{attempt.bytes_received:>16,}  {outcome}",
        flush=True,
    )


def print_summary(attempts: Sequence[Attempt], requested_count: int) -> None:
    """Print aggregate statistics and grouped failure reasons."""
    successful = [attempt for attempt in attempts if attempt.ok]
    failed = [attempt for attempt in attempts if not attempt.ok]
    success_rate = (
        len(successful) * 100.0 / requested_count if requested_count else 0.0
    )

    print()
    print("=" * SUMMARY_WIDTH)
    print("SUMMARY")
    print("=" * SUMMARY_WIDTH)
    print(
        f"{'Successful requests:':<28}"
        f"{len(successful)} / {requested_count} ({success_rate:.1f}%)"
    )
    print(f"{'Failed requests:':<28}{len(failed)}")

    if successful:
        stats = calculate_stats(successful)
        print()
        print("Request timing (successful requests only):")
        print(f"{'  Average:':<28}{stats.average_seconds * 1000:,.2f} ms")
        print(
            f"{'  Min / max:':<28}"
            f"{stats.minimum_seconds * 1000:,.2f} ms / "
            f"{stats.maximum_seconds * 1000:,.2f} ms"
        )

        print()
        print("Downloaded body data:")
        print(f"{'  Total:':<28}{stats.total_bytes:,} bytes")
        print(
            f"{'  Total:':<28}{stats.total_bytes / BYTES_PER_MB:,.2f} MB "
            "(1 MB = 1,000,000 bytes)"
        )

        print()
        print("Average download speed (aggregate throughput):")
        print(
            "  Formula:             "
            "MB/s = sum(bytes) / sum(elapsed time) / 1,000,000"
        )
        print("                       Mbit/s = MB/s x 8")
        print(
            f"  Calculation:         {stats.total_bytes:,} B / "
            f"{stats.total_seconds:.6f} s = {stats.speed_bytes_s:,.3f} B/s"
        )
        print(
            f"  Speed:               {stats.speed_mb_s:,.3f} MB/s = "
            f"{stats.speed_mbit_s:,.3f} Mbit/s"
        )

    if failed:
        reasons = Counter(_attempt_error(attempt) for attempt in failed)
        print()
        print("Failure reasons:")
        for reason, count in reasons.most_common(MAX_FAILURE_REASONS):
            print(f"  {count:>2}x  {reason}")
        hidden_reasons = len(reasons) - MAX_FAILURE_REASONS
        if hidden_reasons > 0:
            print(f"  ... and {hidden_reasons} more distinct error(s)")

    if not successful:
        print()
        print("No successful requests; timing and speed statistics are unavailable.")
    elif failed:
        print()
        print("Note: failed attempts are excluded from timing, bytes and speed.")

    print("=" * SUMMARY_WIDTH)


def run_benchmark(url: str, request_count: int, timeout_seconds: float) -> int:
    """Run all attempts sequentially, print results, and return an exit code."""
    print_header(url, request_count, timeout_seconds)
    attempts: List[Attempt] = []

    for number in range(1, request_count + 1):
        attempt = perform_request(url, timeout_seconds, number)
        attempts.append(attempt)
        print_attempt(attempt, request_count)

    print_summary(attempts, request_count)
    return 0 if all(attempt.ok for attempt in attempts) else 1


def configure_output() -> None:
    """Avoid encoding crashes when output exceeds the legacy console code page."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, OSError, ValueError):
            # Not a TextIOWrapper, for example when embedded in another app.
            pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse arguments, handle user interruption, and run the benchmark."""
    configure_output()
    args = build_parser().parse_args(argv)
    try:
        return run_benchmark(args.url, args.requests, args.timeout)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
