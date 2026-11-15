"""Regression checks for forecast timing, trade timing and transaction costs."""

import unittest

import numpy as np
import pandas as pd

from forecast import add_features, evaluate, max_drawdown, net_intraday_returns
from models import lagged_windows


class ForecastTimingTests(unittest.TestCase):
    def test_window_excludes_return_on_target_day(self):
        features = np.arange(12).reshape(6, 2)
        windows = lagged_windows(features, lookback=3)
        self.assertTrue(np.isnan(windows[:3]).all())
        np.testing.assert_array_equal(windows[3], features[:3])
        np.testing.assert_array_equal(windows[-1], features[2:5])

    def test_regime_at_target_date_is_unchanged_by_target_prices(self):
        dates = pd.date_range("2024-01-01", periods=40)
        prices = pd.DataFrame({"open": 100 + np.arange(40), "close": 100 + np.arange(40)},
                              index=dates)
        before = add_features(prices).iloc[-1]
        prices.loc[dates[-1], "close"] *= 2
        after = add_features(prices).iloc[-1]
        self.assertEqual(before["known_volatility"], after["known_volatility"])
        self.assertEqual(before["known_trend"], after["known_trend"])
        self.assertNotEqual(before["daily_return"], after["daily_return"])

    def test_forecast_and_execution_use_different_price_intervals(self):
        rows = pd.DataFrame({"open": [110.0], "close": [105.0], "previous_close": [100.0],
                             "daily_return": [0.05], "known_volatility": [0.01],
                             "known_trend": [0.1]}, index=pd.to_datetime(["2024-01-01"]))
        forecasts = {"Naive": np.zeros(1), "ARIMA": np.log([1.05])}
        daily, metrics, backtest = evaluate("BTC-USD", rows, forecasts, 0.02, 0.001)
        self.assertAlmostEqual(metrics[0]["rmse_usd"], 5.0)
        self.assertAlmostEqual(metrics[1]["rmse_usd"], 0.0)
        self.assertEqual(daily["naive_net_return"].iloc[0], 0.0)
        self.assertAlmostEqual(daily["arima_net_return"].iloc[0], 105 / 110 * 0.999**2 - 1)
        self.assertEqual(backtest[2]["close_directional_accuracy"], 1.0)
        self.assertEqual(backtest[2]["intraday_directional_accuracy"], 0.0)

    def test_two_fills_and_first_day_drawdown(self):
        net = net_intraday_returns(np.array([True, False]), np.array([0.0, 0.2]), 0.001)
        np.testing.assert_allclose(net, [0.999**2 - 1, 0.0])
        self.assertAlmostEqual(max_drawdown(net), net[0])


if __name__ == "__main__":
    unittest.main()
