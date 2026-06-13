import contextlib
import io
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from benchmarks.result_logger import load_result_by_id, run_benchmark_from_result


class ResultLoggerReplaySmokeTests(unittest.TestCase):
    def test_load_result_by_id_and_replay_from_fixture(self):
        fixture_dir = Path(__file__).parent / "fixtures"
        result = load_result_by_id("sample_run", results_dir=str(fixture_dir))
        prices = pd.DataFrame(
            {
                "AAA": [100.0, 101.0, 102.0],
                "BBB": [100.0, 99.0, 101.0],
            },
            index=pd.date_range("2025-01-01", periods=3, freq="D"),
        )

        with (
            patch("benchmarks.result_logger.load_oos_prices_for_period", return_value=prices) as load_prices,
            patch("benchmarks.result_logger.plot_oos_comparison") as plot_oos,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            run_benchmark_from_result(result)

        load_prices.assert_called_once_with(["AAA", "BBB"], "2025-01-01 to 2025-01-05")
        plot_oos.assert_called_once()
