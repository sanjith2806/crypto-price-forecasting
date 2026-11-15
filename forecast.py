"""Run the BTC-USD and ETH-USD one-day-ahead forecasting experiment."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models import LOOKBACK, arima_forecasts, lstm_forecasts

ASSETS = {"BTC-USD": "btc", "ETH-USD": "eth"}
START = "2018-01-01"
VALIDATION_START = "2023-01-01"
TEST_START = "2024-01-01"
END = "2026-02-07"
MODEL_NAMES = ("Naive", "ARIMA", "LSTM")
REGIMES = ("Quiet up", "Quiet down", "High-vol up", "High-vol down")
MARKET = "Always long"


def load_prices(data_dir, ticker):
    """Load earlier project exports or fetch the same Yahoo Finance daily fields."""
    if data_dir is None:
        import yfinance as yf

        end_exclusive = (pd.Timestamp(END) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        prices = yf.download(ticker, start=START, end=end_exclusive, interval="1d",
                             auto_adjust=False, progress=False, multi_level_index=False)
    else:
        processed = data_dir / f"{ASSETS[ticker]}_processed.csv"
        raw = data_dir / f"{ticker.replace('-', '_')}_daily.csv"
        source = next((path for path in (processed, data_dir / "processed" / processed.name, raw)
                       if path.is_file()), None)
        if source is None:
            raise FileNotFoundError(f"{ticker}: expected {processed.name} or {raw.name} in {data_dir}")
        prices = pd.read_csv(source).dropna(subset=["Date"]).set_index("Date")

    if prices.empty:
        raise ValueError(f"{ticker}: no price data returned")
    dates = pd.to_datetime(prices.index, utc=True).tz_convert(None).normalize()
    prices = prices[["Open", "Adj Close"]].apply(pd.to_numeric).set_axis(dates).sort_index()
    prices = prices.loc[START:END]
    expected_dates = pd.date_range(START, END, name=prices.index.name)
    if not prices.index.equals(expected_dates):
        raise ValueError(f"{ticker}: expected one observation per calendar day from {START} to {END}")
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError(f"{ticker}: prices must be finite and positive")
    # For these spot crypto series Open and Adj Close are in the same USD units.
    return prices.rename(columns={"Open": "open", "Adj Close": "close"})


def add_features(prices):
    rows = prices.copy()
    rows["previous_close"] = rows["close"].shift(1)
    rows["daily_return"] = rows["close"].pct_change()
    rows["log_return"] = np.log(rows["close"]).diff()
    rows["volatility_30d"] = rows["daily_return"].rolling(30).std()
    # Regime labels belong to the target day, so shift both indicators by a day.
    rows["known_volatility"] = rows["volatility_30d"].shift(1)
    rows["known_trend"] = rows["close"].pct_change(30).shift(1)
    return rows


def model_partitions(rows):
    dates = rows.index
    eligible = np.arange(len(rows)) >= 30 + LOOKBACK
    training = eligible & (dates < VALIDATION_START)
    validation = (dates >= VALIDATION_START) & (dates < TEST_START)
    test = dates >= TEST_START
    return training, validation, test


def assign_regimes(test_rows, cutoff):
    return np.char.add(
        np.where(test_rows["known_volatility"] > cutoff, "High-vol ", "Quiet "),
        np.where(test_rows["known_trend"] >= 0, "up", "down"),
    )


def net_intraday_returns(position, intraday_return, fee):
    """A long trade uses the next day's open and close and pays fees at both fills."""
    return np.where(position, (1 + intraday_return) * (1 - fee) ** 2 - 1, 0.0)


def directional_accuracy(prediction, actual):
    """A zero forecast is a flat call and only matches a flat realised return."""
    return float(np.mean(np.sign(prediction) == np.sign(actual)))


def max_drawdown(returns):
    equity = np.r_[1.0, np.cumprod(1 + returns)]
    return float(np.min(equity / np.maximum.accumulate(equity) - 1))


def evaluate(ticker, rows, log_predictions, cutoff, fee):
    predictions = rows[["open", "close", "previous_close", "daily_return",
                        "known_volatility", "known_trend"]].copy()
    predictions.index.name = "target_date"
    predictions["asset"] = ticker
    predictions["regime"] = assign_regimes(predictions, cutoff)
    predictions["intraday_return"] = predictions["close"] / predictions["open"] - 1
    model_metrics = []
    regime_metrics = []
    for model, predicted_log_return in log_predictions.items():
        name = model.lower().replace(" ", "_")
        signal = np.sign(predicted_log_return)
        position = signal > 0
        returns = net_intraday_returns(position, predictions["intraday_return"], fee)
        predictions[f"{name}_signal"] = signal.astype(int)
        predictions[f"{name}_net_return"] = returns
        if model in MODEL_NAMES:
            estimate = predictions["previous_close"] * np.exp(predicted_log_return)
            predictions[f"{name}_close"] = estimate
            error = estimate - predictions["close"]
            model_metrics.append({"asset": ticker, "model": model, "days": len(rows),
                                  "rmse_usd": float(np.sqrt(np.mean(error ** 2))),
                                  "mae_usd": float(np.mean(np.abs(error)))})
        for regime in ("All", *REGIMES):
            mask = np.ones(len(rows), dtype=bool) if regime == "All" else predictions["regime"] == regime
            if not mask.any():
                continue
            subset = predictions.loc[mask]
            regime_rows = {"asset": ticker, "model": model, "regime": regime, "days": len(subset),
                           "close_directional_accuracy": directional_accuracy(signal[mask], subset["daily_return"]),
                           "intraday_directional_accuracy": directional_accuracy(signal[mask], subset["intraday_return"]),
                           "long_trades": int(position[mask].sum()),
                           "compounded_net_return": float(np.prod(1 + returns[mask]) - 1)}
            if regime == "All":
                regime_rows["max_drawdown"] = max_drawdown(returns)
            regime_metrics.append(regime_rows)
    return predictions, model_metrics, regime_metrics


def plot_results(predictions, metrics, backtest, output_dir, fee_bps):
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    colours = {"Naive": "#788391", "ARIMA": "#007d75", "LSTM": "#7d529b", MARKET: "#bd6a20"}
    for column, ticker in enumerate(ASSETS):
        errors = metrics[metrics["asset"] == ticker].set_index("model")
        relative_rmse = 100 * errors["rmse_usd"] / errors.loc["Naive", "rmse_usd"]
        bars = axes[0, column].bar(relative_rmse.index, relative_rmse, color=[colours[m] for m in MODEL_NAMES])
        axes[0, column].axhline(100, color="#343f4a", linewidth=1, linestyle="--")
        axes[0, column].bar_label(bars, fmt="%.2f%%", padding=3)
        axes[0, column].set_ylim(0, max(115, relative_rmse.max() * 1.12))
        axes[0, column].set(title=f"{ticker} · Price forecast error", ylabel="RMSE (% of naïve)")
        data = predictions[predictions["asset"] == ticker]
        for model in (MARKET, *MODEL_NAMES):
            name = model.lower().replace(" ", "_")
            equity = (1 + data[f"{name}_net_return"]).cumprod()
            axes[1, column].plot(equity.index, equity, label=model, color=colours[model], linewidth=1.35)
        axes[1, column].axhline(1.0, color="#343f4a", linewidth=0.7, linestyle="--")
        days = int(backtest.loc[(backtest["asset"] == ticker) & (backtest["regime"] == "All"), "days"].iloc[0])
        axes[1, column].set(title=f"Open-to-close backtest · {days} days", ylabel="Growth of $1")
        axes[1, column].legend(frameon=False, fontsize=8, ncol=2)
    figure.suptitle(f"Bitcoin and Ethereum · Held-out results · {fee_bps:g} bps per fill", fontsize=15)
    figure.savefig(output_dir / "results.png", dpi=155)
    plt.close(figure)


def run(data_dir, output_dir, fee_bps):
    if not 0 <= fee_bps < 10_000:
        raise ValueError("Fee must be at least 0 and below 10,000 basis points")
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions, price_metrics, regime_metrics = [], [], []
    settings = {"data_start": START, "validation_start": VALIDATION_START,
                "test_start": TEST_START, "data_end": END, "fee_bps_per_fill": fee_bps, "assets": {}}
    for ticker in ASSETS:
        print(f"Fitting {ticker}", flush=True)
        rows = add_features(load_prices(data_dir, ticker))
        train, validation, test = model_partitions(rows)
        arima, arima_settings = arima_forecasts(rows, train, validation, test)
        lstm, lstm_settings = lstm_forecasts(rows, train, validation, test)
        cutoff = float(rows.loc[train | validation, "known_volatility"].median())
        settings["assets"][ticker] = {"arima": arima_settings, "lstm": lstm_settings,
                                      "volatility_cutoff": cutoff}
        forecasts = {"Naive": np.zeros(test.sum()), "ARIMA": arima, "LSTM": lstm, MARKET: np.ones(test.sum())}
        detail, metrics, regimes = evaluate(ticker, rows.loc[test], forecasts, cutoff, fee_bps / 10_000)
        predictions.append(detail)
        price_metrics.extend(metrics)
        regime_metrics.extend(regimes)

    daily = pd.concat(predictions)
    prices = pd.DataFrame(price_metrics)
    backtest = pd.DataFrame(regime_metrics)
    daily.to_csv(output_dir / "predictions.csv", float_format="%.10g")
    prices.to_csv(output_dir / "forecast_metrics.csv", index=False, float_format="%.10g")
    backtest.to_csv(output_dir / "backtest_metrics.csv", index=False, float_format="%.10g")
    (output_dir / "model_settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    plot_results(daily, prices, backtest, output_dir, fee_bps)
    print(prices.to_string(index=False))
    print(f"Saved evaluation to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, help="Directory containing the earlier analysis CSV exports")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--fee-bps", type=float, default=10, help="Fee in basis points per fill (default 10)")
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, args.fee_bps)
