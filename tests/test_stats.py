"""Unit tests for httpbench.stats."""

import math
import unittest

from httpbench.network import Attempt, Phases
from httpbench.stats import calculate_stats, percentile


def make_attempt(
    number: int,
    elapsed: float,
    body_bytes: int,
    *,
    status: int | None = 200,
    error: str | None = None,
    dns: float | None = None,
    connect: float | None = None,
    tls: float | None = None,
    ttfb: float | None = None,
    body: float | None = None,
) -> Attempt:
    return Attempt(
        number=number,
        status=status,
        elapsed_seconds=elapsed,
        bytes_received=body_bytes,
        phases=Phases(
            dns_seconds=dns,
            connect_seconds=connect,
            tls_seconds=tls,
            ttfb_seconds=ttfb,
            body_seconds=body,
        ),
        error=error,
    )


class PercentileTests(unittest.TestCase):
    def test_linear_interpolation(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(percentile(values, 90), 3.7)
        self.assertAlmostEqual(percentile(values, 95), 3.85)
        self.assertAlmostEqual(percentile(values, 50), 2.5)

    def test_single_value(self) -> None:
        self.assertEqual(percentile([2.5], 90), 2.5)

    def test_rejects_empty_input(self) -> None:
        with self.assertRaises(ValueError):
            percentile([], 90)

    def test_rejects_bad_percent(self) -> None:
        with self.assertRaises(ValueError):
            percentile([1.0], 0)
        with self.assertRaises(ValueError):
            percentile([1.0], 100)


class CalculateStatsTests(unittest.TestCase):
    def test_basic_aggregates(self) -> None:
        attempts = [
            make_attempt(1, 1.0, 1_000_000, body=0.9),
            make_attempt(2, 3.0, 3_000_000, body=2.9),
        ]

        stats = calculate_stats(attempts)

        self.assertEqual(stats.count, 2)
        self.assertEqual(stats.total_bytes, 4_000_000)
        self.assertAlmostEqual(stats.total_seconds, 4.0)
        self.assertAlmostEqual(stats.average_seconds, 2.0)
        self.assertAlmostEqual(stats.median_seconds, 2.0)
        self.assertAlmostEqual(stats.minimum_seconds, 1.0)
        self.assertAlmostEqual(stats.maximum_seconds, 3.0)
        self.assertAlmostEqual(stats.speed_bytes_s, 1_000_000.0)
        self.assertAlmostEqual(stats.speed_mb_s, 1.0)
        self.assertAlmostEqual(stats.speed_mbit_s, 8.0)
        self.assertAlmostEqual(stats.body_seconds, 3.8)
        self.assertAlmostEqual(stats.body_speed_bytes_s, 4_000_000 / 3.8)
        self.assertAlmostEqual(stats.body_average_seconds, 1.9)

    def test_median_stddev_and_percentiles(self) -> None:
        attempts = [
            make_attempt(number, elapsed, 1_000)
            for number, elapsed in enumerate((1.0, 2.0, 3.0, 4.0), start=1)
        ]

        stats = calculate_stats(attempts)

        self.assertAlmostEqual(stats.median_seconds, 2.5)
        self.assertAlmostEqual(stats.std_dev_seconds, math.sqrt(5.0 / 3.0))
        self.assertAlmostEqual(stats.p90_seconds, 3.7)
        self.assertAlmostEqual(stats.p95_seconds, 3.85)

    def test_single_attempt_has_zero_stddev(self) -> None:
        stats = calculate_stats([make_attempt(1, 1.0, 100)])
        self.assertEqual(stats.std_dev_seconds, 0.0)
        self.assertAlmostEqual(stats.p90_seconds, 1.0)

    def test_phase_averages_skip_reused_attempts(self) -> None:
        attempts = [
            make_attempt(
                1, 1.0, 1_000, dns=0.010, connect=0.020, tls=0.030, ttfb=0.100
            ),
            make_attempt(2, 0.5, 1_000, dns=None, connect=None, tls=None, ttfb=0.060),
            make_attempt(3, 0.4, 1_000, dns=0.011, connect=0.019, tls=None, ttfb=0.040),
        ]

        stats = calculate_stats(attempts)

        self.assertAlmostEqual(stats.dns_seconds, 0.0105)
        self.assertAlmostEqual(stats.connect_seconds, 0.0195)
        self.assertAlmostEqual(stats.tls_seconds, 0.030)
        self.assertAlmostEqual(stats.ttfb_seconds, (0.100 + 0.060 + 0.040) / 3)

    def test_phase_averages_none_when_never_measured(self) -> None:
        attempts = [make_attempt(1, 1.0, 1_000, dns=None, connect=None, tls=None)]

        stats = calculate_stats(attempts)

        self.assertIsNone(stats.dns_seconds)
        self.assertIsNone(stats.connect_seconds)
        self.assertIsNone(stats.tls_seconds)

    def test_zero_body_time_gives_zero_body_speed(self) -> None:
        stats = calculate_stats([make_attempt(1, 1.0, 100, body=0.0)])
        self.assertEqual(stats.body_speed_bytes_s, 0.0)

    def test_empty_input_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            calculate_stats([])

    def test_zero_total_time_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            calculate_stats([make_attempt(1, 0.0, 100)])


class AttemptTests(unittest.TestCase):
    def test_ok_requires_2xx_without_error(self) -> None:
        self.assertTrue(make_attempt(1, 1.0, 10).ok)
        self.assertTrue(make_attempt(1, 1.0, 10, status=204).ok)
        self.assertFalse(make_attempt(1, 1.0, 10, status=404).ok)
        self.assertFalse(make_attempt(1, 1.0, 10, status=None).ok)
        self.assertFalse(make_attempt(1, 1.0, 10, status=200, error="timeout").ok)


if __name__ == "__main__":
    unittest.main()
