# Cryptocurrency Price Forecasting Model
Comparing ARIMA and LSTM models for Bitcoin and Ethereum price prediction using Python and PyTorch.

This project compares a classical statistical approach (ARIMA) against a deep learning approach (LSTM) for forecasting daily cryptocurrency prices. The goal is to evaluate whether increased model complexity translates to improved predictive accuracy, using RMSE and MAE against a naïve baseline.

## Project Structure
crypto-price-forecasting/
├── data/
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_stationarity_tests.ipynb
│   ├── 03_arima_model.ipynb
│   ├── 04_lstm_model.ipynb
│   └── 05_evaluation.ipynb
├── README.md
└── requirements.txt

## Technologies
Python, Pandas, NumPy, PyTorch, statsmodels, scikit-learn, Matplotlib

## Status
🚧 In progress — data loading and stationarity testing complete. ARIMA and LSTM implementation in progress.
