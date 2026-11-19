"""Unit tests for httpbench.report (all writers take a stream)."""

import csv
import io
import json
import unittest

from httpbench.network import Attempt, Phases
from httpbench.report import (
    CSV_COLUMNS,
    print_attempt,
    print_run_header,
    print_summary,
    print_warmup_attempt,
    write_csv_report,
    write_json_report,
)


def make_attempt(
    number: int,
    status: int | None = 200,
    elapsed: float = 1.0,
    received: int = 1_000_000,
    error: str | None = None,
) -> Attempt:
    return Attempt(
        number=number,
        status=status,
        elapsed_seconds=elapsed,
        bytes_received=received,
        phases=Phases(
            dns_seconds=0.001,
            connect_seconds=0.002,
            tls_seconds=None,
            ttfb_seconds=0.1,
            body_seconds=0.8,
        ),
        error=error,
    )


class SummaryTests(unittest.TestCase):
    def test_groups_failure_reasons(self) -> None:
        attempts = [
            make_attempt(1),
            make_attempt(
                2, status=404, elapsed=0.1, received=0, error="HTTP 404 Not Found"
            ),
            make_attempt(
                3, status=404, elapsed=0.1, received=0, error="HTTP 404 Not Found"
            ),
        ]
        output = io.StringIO()

        print_summary(attempts, 3, stream=output)

        text = output.getvalue()
        self.assertIn("1 / 3 (33.3%)", text)
        self.assertIn("Failure reasons:", text)
        self.assertIn("2x  HTTP 404 Not Found", text)

    def test_timing_rows_include_median_percentiles_and_stddev(self) -> None:
        attempts = [make_attempt(number, elapsed=number / 100) for number in (1, 2, 3)]
        output = io.StringIO()

        print_summary(attempts, 3, stream=output)

        text = output.getvalue()
        self.assertIn("Median:", text)
        self.assertIn("Std dev:", text)
        self.assertIn("p90 / p95:", text)
        self.assertIn("Min / max:", text)

    def test_total_appears_once_with_bytes_and_megabytes(self) -> None:
        attempts = [make_attempt(1)]
        output = io.StringIO()

        print_summary(attempts, 1, stream=output)

        text = output.getvalue()
        self.assertEqual(text.count("Total:"), 1)
        self.assertIn("1,000,000 bytes (1.00 MB)", text)

    def test_interrupted_summary_is_labelled(self) -> None:
        attempts = [make_attempt(1)]
        output = io.StringIO()

        print_summary(attempts, 10, interrupted=True, stream=output)

        self.assertIn("SUMMARY (interrupted after 1 of 10)", output.getvalue())

    def test_no_successful_attempts_message(self) -> None:
        attempts = [make_attempt(1, status=None, error="network error: boom")]
        output = io.StringIO()

        print_summary(attempts, 1, stream=output)

        self.assertIn("No successful requests", output.getvalue())

    def test_both_speeds_are_reported(self) -> None:
        attempts = [make_attempt(1, elapsed=1.0, received=1_000_000)]
        output = io.StringIO()

        print_summary(attempts, 1, stream=output)

        text = output.getvalue()
        self.assertIn("Body transfer speed", text)
        self.assertIn("Aggregate request speed", text)


class ProgressRowTests(unittest.TestCase):
    def test_attempt_row_alignment(self) -> None:
        output = io.StringIO()

        print_attempt(make_attempt(1, error="boom"), 10, stream=output)

        line = output.getvalue().rstrip("\n")
        self.assertTrue(line.startswith("[1/10]"))
        self.assertIn("ERROR: boom", line)

    def test_warmup_row_label(self) -> None:
        output = io.StringIO()

        print_warmup_attempt(make_attempt(2), 3, stream=output)

        self.assertIn("[warmup 2/3]", output.getvalue())

    def test_run_header_lists_non_default_options(self) -> None:
        output = io.StringIO()

        print_run_header(
            "http://example.com/f",
            5,
            12.5,
            warmup_count=2,
            keep_alive=True,
            delay_seconds=0.5,
            cache_bust=True,
            stream=output,
        )

        text = output.getvalue()
        self.assertIn("http://example.com/f", text)
        self.assertIn("Warmup:   2", text)
        self.assertIn("Keep-alive:", text)
        self.assertIn("Delay:    0.5 s", text)
        self.assertIn("Cache busting: on", text)

    def test_run_header_hides_default_options(self) -> None:
        output = io.StringIO()

        print_run_header("http://example.com/f", 5, 30, stream=output)

        text = output.getvalue()
        self.assertNotIn("Warmup", text)
        self.assertNotIn("Keep-alive", text)


class JsonReportTests(unittest.TestCase):
    def test_json_structure(self) -> None:
        attempts = [
            make_attempt(1),
            make_attempt(2, status=500, error="HTTP 500 oops"),
        ]
        output = io.StringIO()

        write_json_report("http://example.com/f", attempts, 2, False, stream=output)

        data = json.loads(output.getvalue())
        self.assertEqual(data["url"], "http://example.com/f")
        self.assertEqual(data["requested_requests"], 2)
        self.assertEqual(data["completed_requests"], 2)
        self.assertFalse(data["interrupted"])
        self.assertEqual(len(data["attempts"]), 2)

        first = data["attempts"][0]
        self.assertTrue(first["ok"])
        self.assertEqual(first["status"], 200)
        self.assertEqual(first["bytes_received"], 1_000_000)
        phases = first["phases_seconds"]
        self.assertAlmostEqual(phases["dns"], 0.001)
        self.assertIsNone(phases["tls"])

        second = data["attempts"][1]
        self.assertFalse(second["ok"])
        self.assertEqual(second["error"], "HTTP 500 oops")

        summary = data["summary"]
        self.assertEqual(summary["count"], 1)
        self.assertGreater(summary["speed_bytes_s"], 0.0)

    def test_json_summary_is_null_without_successes(self) -> None:
        attempts = [make_attempt(1, status=None, error="DNS error: nope")]
        output = io.StringIO()

        write_json_report("http://example.com/f", attempts, 1, True, stream=output)

        data = json.loads(output.getvalue())
        self.assertTrue(data["interrupted"])
        self.assertIsNone(data["summary"])


class CsvReportTests(unittest.TestCase):
    def test_csv_rows_and_columns(self) -> None:
        attempts = [
            make_attempt(1),
            make_attempt(2, status=404, received=0, error="HTTP 404 Not Found"),
        ]
        output = io.StringIO(newline="")

        write_csv_report(attempts, stream=output)

        rows = list(csv.reader(io.StringIO(output.getvalue())))
        self.assertEqual(rows[0], list(CSV_COLUMNS))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[1][0], "1")
        self.assertEqual(rows[1][1], "True")
        self.assertEqual(rows[2][1], "False")
        self.assertEqual(rows[2][7], "HTTP 404 Not Found")
        self.assertAlmostEqual(float(rows[1][8]), 0.001)


if __name__ == "__main__":
    unittest.main()
