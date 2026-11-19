"""Aggregated statistics over benchmark attempts."""

import math
import statistics
from dataclasses import dataclass

from httpbench.network import Attempt

# Decimal (SI) units: 1 MB = 1,000,000 bytes, 1 byte = 8 bits.
BYTES_PER_MB = 1_000_000
BITS_PER_BYTE = 8


def percentile(values: list[float], percent: int) -> float:
    """Percentile with linear interpolation (NumPy's default method)."""
    if not values:
        raise ValueError("values must not be empty")
    if not 1 <= percent <= 99:
        raise ValueError("percent must be between 1 and 99")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    cuts = statistics.quantiles(ordered, n=100, method="inclusive")
    return cuts[percent - 1]


def _phase_average(attempts: list[Attempt], phase: str) -> float | None:
    """Average one phase over the attempts where it was actually measured.

    DNS, TCP and TLS are absent on reused keep-alive connections, so those
    attempts do not dilute the average with zeros.
    """
    values = [
        value
        for attempt in attempts
        if (value := getattr(attempt.phases, phase)) is not None
    ]
    if not values:
        return None
    return math.fsum(values) / len(values)


@dataclass(frozen=True)
class PerformanceStats:
    """Aggregate statistics for successful attempts."""

    count: int
    total_bytes: int
    total_seconds: float
    average_seconds: float
    median_seconds: float
    std_dev_seconds: float
    minimum_seconds: float
    maximum_seconds: float
    p90_seconds: float
    p95_seconds: float
    # Total of the body transfer windows (last body byte - first body byte).
    body_seconds: float
    # Phase averages, None when the phase never happened (keep-alive reuse).
    dns_seconds: float | None
    connect_seconds: float | None
    tls_seconds: float | None
    ttfb_seconds: float | None

    @property
    def speed_bytes_s(self) -> float:
        """Aggregate throughput: total bytes over total request time."""
        return self.total_bytes / self.total_seconds

    @property
    def speed_mb_s(self) -> float:
        return self.speed_bytes_s / BYTES_PER_MB

    @property
    def speed_mbit_s(self) -> float:
        return self.speed_mb_s * BITS_PER_BYTE

    @property
    def body_speed_bytes_s(self) -> float:
        """Body throughput: body bytes over the body transfer windows only."""
        if self.body_seconds <= 0:
            return 0.0
        return self.total_bytes / self.body_seconds

    @property
    def body_speed_mb_s(self) -> float:
        return self.body_speed_bytes_s / BYTES_PER_MB

    @property
    def body_speed_mbit_s(self) -> float:
        return self.body_speed_mb_s * BITS_PER_BYTE

    @property
    def body_average_seconds(self) -> float:
        """Average body transfer window per successful attempt."""
        return self.body_seconds / self.count if self.count else 0.0


def calculate_stats(attempts: list[Attempt]) -> PerformanceStats:
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
        average_seconds=total_seconds / len(attempts),
        median_seconds=statistics.median(times),
        std_dev_seconds=statistics.stdev(times) if len(times) > 1 else 0.0,
        minimum_seconds=min(times),
        maximum_seconds=max(times),
        p90_seconds=percentile(times, 90),
        p95_seconds=percentile(times, 95),
        body_seconds=math.fsum(
            attempt.phases.body_seconds or 0.0 for attempt in attempts
        ),
        dns_seconds=_phase_average(attempts, "dns_seconds"),
        connect_seconds=_phase_average(attempts, "connect_seconds"),
        tls_seconds=_phase_average(attempts, "tls_seconds"),
        ttfb_seconds=_phase_average(attempts, "ttfb_seconds"),
    )
