"""One-step ARIMA and LSTM forecasts of daily log returns."""

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch import nn

ARIMA_LAGS = (0, 1, 2, 3, 5, 7)
LOOKBACK = 30
EPOCHS = (5, 10, 20, 40)
SEED = 42


def price_rmse(log_return, rows):
    """Use USD price RMSE to select both models on the same validation dates."""
    predicted_close = rows["previous_close"].to_numpy() * np.exp(log_return)
    return float(np.sqrt(np.mean((predicted_close - rows["close"].to_numpy()) ** 2)))


def arima_forecasts(rows, training, validation, test):
    """Fit ARIMA(p,1,0) with drift on log price by OLS on its first differences."""
    log_return = rows["log_return"]
    lags = pd.concat([log_return.shift(lag) for lag in range(1, max(ARIMA_LAGS) + 1)], axis=1)
    design = np.column_stack([np.ones(len(rows)), lags.to_numpy()])
    target = log_return.to_numpy()

    def fit(lag_count, sample):
        return np.linalg.lstsq(design[sample, :lag_count + 1], target[sample], rcond=None)[0]

    scores = {}
    for lag_count in ARIMA_LAGS:
        coefficients = fit(lag_count, training)
        predicted = design[validation, :lag_count + 1] @ coefficients
        scores[lag_count] = price_rmse(predicted, rows.loc[validation])
    order = min(scores, key=scores.get)
    coefficients = fit(order, training | validation)
    predictions = design[test, :order + 1] @ coefficients
    settings = {"order": [order, 1, 0], "drift": float(coefficients[0]),
                "ar_coefficients": coefficients[1:].tolist(), "validation_rmse": scores}
    return predictions, settings


class ReturnLSTM(nn.Module):
    def __init__(self, feature_count):
        super().__init__()
        self.lstm = nn.LSTM(feature_count, hidden_size=16, batch_first=True)
        self.output = nn.Linear(16, 1)

    def forward(self, windows):
        states, _ = self.lstm(windows)
        return self.output(states[:, -1, :]).squeeze(-1)


def lagged_windows(features, lookback):
    """The window attached to target day t ends at t-1."""
    windows = np.full((len(features), lookback, features.shape[1]), np.nan)
    windows[lookback:] = sliding_window_view(features[:-1], lookback, axis=0).swapaxes(1, 2)
    return windows


def lstm_forecasts(rows, training, validation, test):
    """Select a training duration on validation, then refit through that period."""
    features = rows[["log_return", "volatility_30d"]].to_numpy()
    windows = lagged_windows(features, LOOKBACK)
    target = rows["log_return"].to_numpy()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    def fit_and_predict(fit_mask, prediction_mask, checkpoints):
        torch.manual_seed(SEED)
        # Only feature windows and targets from the fitting sample set the scales.
        fit_windows = windows[fit_mask]
        feature_mean = fit_windows.mean(axis=(0, 1))
        feature_std = np.maximum(fit_windows.std(axis=(0, 1)), 1e-8)
        target_mean = target[fit_mask].mean()
        target_std = max(target[fit_mask].std(), 1e-8)
        train_x = torch.tensor((fit_windows - feature_mean) / feature_std, dtype=torch.float32)
        train_y = torch.tensor((target[fit_mask] - target_mean) / target_std, dtype=torch.float32)
        predict_x = torch.tensor((windows[prediction_mask] - feature_mean) / feature_std,
                                 dtype=torch.float32)

        model = ReturnLSTM(features.shape[1])
        optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_function = nn.MSELoss()
        predictions = {}
        for epoch in range(1, max(checkpoints) + 1):
            model.train()
            for batch in torch.randperm(len(train_x)).split(64):
                optimiser.zero_grad()
                loss = loss_function(model(train_x[batch]), train_y[batch])
                loss.backward()
                optimiser.step()
            if epoch in checkpoints:
                model.eval()
                with torch.inference_mode():
                    predictions[epoch] = model(predict_x).numpy().astype(float) * target_std + target_mean
        return predictions

    candidates = fit_and_predict(training, validation, EPOCHS)
    scores = {epoch: price_rmse(predictions, rows.loc[validation])
              for epoch, predictions in candidates.items()}
    best_epoch = min(scores, key=scores.get)
    final = fit_and_predict(training | validation, test, (best_epoch,))
    return final[best_epoch], {"epochs": best_epoch, "validation_rmse": scores, "seed": SEED}
