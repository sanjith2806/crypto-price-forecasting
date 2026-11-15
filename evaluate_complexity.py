"""Measure model runtime and bootstrap the held-out RMSE differences."""

import argparse
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import pandas as pd

from forecast import ASSETS, add_features, load_prices, model_partitions
from models import arima_forecasts, lstm_forecasts


def benchmark(data_dir, repeats):
    records = []
    for asset in ASSETS:
        rows = add_features(load_prices(data_dir, asset))
        partitions = model_partitions(rows)
        test = partitions[-1]

        def forecast_naive():
            return rows.loc[test, "previous_close"].to_numpy(copy=True)

        def forecast_model(fit_and_predict):
            log_returns, _ = fit_and_predict(rows, *partitions)
            return rows.loc[test, "previous_close"].to_numpy() * np.exp(log_returns)

        for name, forecast, runs in (
            ("Naive", forecast_naive, 101),
            ("ARIMA", lambda: forecast_model(arima_forecasts), repeats),
            ("LSTM", lambda: forecast_model(lstm_forecasts), repeats),
        ):
            elapsed = []
            for _ in range(runs):
                start = perf_counter()
                forecast()
                elapsed.append(perf_counter() - start)
            records.append({"asset": asset, "model": name, "seconds": median(elapsed), "runs": runs})
    return pd.DataFrame(records)


def rmse_intervals(predictions):
    """Resample aligned daily errors in circular blocks with a fixed seed."""
    rng = np.random.default_rng(42)
    records = []
    for asset, rows in predictions.groupby("asset", sort=False):
        actual = rows["close"].to_numpy()
        naive_squared = (rows["naive_close"].to_numpy() - actual) ** 2
        count = len(rows)
        for block_days in (7, 14, 30):
            block_count = (count + block_days - 1) // block_days
            starts = rng.integers(count, size=(10_000, block_count))
            samples = ((starts[:, :, None] + np.arange(block_days)) % count).reshape(10_000, -1)[:, :count]
            naive_means = naive_squared[samples].mean(axis=1)
            for model in ("ARIMA", "LSTM"):
                model_squared = (rows[f"{model.lower()}_close"].to_numpy() - actual) ** 2
                change = 100 * (np.sqrt(model_squared.mean() / naive_squared.mean()) - 1)
                resampled_change = 100 * (np.sqrt(model_squared[samples].mean(axis=1) / naive_means) - 1)
                low, high = np.quantile(resampled_change, [0.025, 0.975])
                records.append({"asset": asset, "model": model, "block_days": block_days,
                                "rmse_change_pct": change, "lower_pct": low, "upper_pct": high})
    return pd.DataFrame(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, help="Directory containing the earlier analysis CSV exports")
    parser.add_argument("--predictions", type=Path, default=Path("results/predictions.csv"))
    parser.add_argument("--repeats", type=int, default=3, help="Number of timing runs per fitted model")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    print("Held-out RMSE change vs naïve (%), with paired 95% bootstrap intervals")
    print(rmse_intervals(pd.read_csv(args.predictions)).to_string(index=False), flush=True)
    print("\nMedian wall time for model selection, refit and held-out forecasts")
    print(benchmark(args.data_dir, args.repeats).to_string(index=False))
