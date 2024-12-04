import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv(r"E:\Projects\walmart_analytics\Cleaned Data\cleaned_data.csv")
df.dtypes

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])
df.set_index("date", inplace=True)

"""**Resample Data to Monthly Frequency**

"""

monthly_data = df["total_price"].resample("ME").sum()

if monthly_data.isnull().sum() > 0:
    print(
        "There are missing values in the resampled data. Filling with forward fill method."
    )
    monthly_data.fillna(method="ffill", inplace=True)

print(monthly_data.shape)

print(monthly_data.index.min())  # First month
print(monthly_data.index.max())  # Last month

result = adfuller(monthly_data)
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")

plt.figure(figsize=(12, 6))
plt.plot(monthly_data, label="Monthly Total Price")
plt.title("Monthly Total Price")
plt.legend()
plt.show()

""" Custom Diagnostics Function


"""


def custom_diagnostics(residuals):
    """Custom diagnostics for small sample sizes"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Residuals over time
    axes[0, 0].plot(residuals)
    axes[0, 0].set_title("Residuals over Time")
    axes[0, 0].axhline(y=0, color="r", linestyle="--")

    # Histogram of residuals
    sns.histplot(residuals, kde=True, ax=axes[0, 1])
    axes[0, 1].set_title("Residuals Distribution")

    # ACF plot
    acf_values = acf(residuals, nlags=min(20, len(residuals) // 2))
    axes[1, 0].plot(range(len(acf_values)), acf_values)
    axes[1, 0].axhline(y=0, color="r", linestyle="--")
    axes[1, 0].axhline(y=1.96 / np.sqrt(len(residuals)), linestyle="--", color="gray")
    axes[1, 0].axhline(y=-1.96 / np.sqrt(len(residuals)), linestyle="--", color="gray")
    axes[1, 0].set_title("ACF of Residuals")

    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("Q-Q Plot")

    plt.tight_layout()
    plt.show()

    # Statistical tests
    print("\nDiagnostic Tests:")
    _, p_value = stats.shapiro(residuals)
    print(f"Shapiro-Wilk test p-value: {p_value:.4f}")

    lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
    print(f"Ljung-Box test p-value: {lb_test['lb_pvalue'].values[0]:.4f}")

    print(f"\nResiduals Statistics:")
    print(f"Mean of residuals: {np.mean(residuals):.4f}")
    print(f"Std of residuals: {np.std(residuals):.4f}")
    print(f"Skewness: {stats.skew(residuals):.4f}")
    print(f"Kurtosis: {stats.kurtosis(residuals):.4f}")


def train_and_evaluate_sarima(data, test_size=0.2):
    # Split data
    train_size = int(len(data) * (1 - test_size))
    train = data[:train_size]
    test = data[train_size:]

    decomposition = seasonal_decompose(data, period=12)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))
    decomposition.observed.plot(ax=ax1)
    ax1.set_title("Original")
    decomposition.trend.plot(ax=ax2)
    ax2.set_title("Trend")
    decomposition.seasonal.plot(ax=ax3)
    ax3.set_title("Seasonal")
    decomposition.resid.plot(ax=ax4)
    ax4.set_title("Residual")
    plt.tight_layout()
    plt.show()

    result = adfuller(data)
    print("Stationarity Test:")
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    print("Critical values:")
    for key, value in result[4].items():
        print(f"\t{key}: {value}")

    order = (2, 1, 1)  # (p, d, q)
    seasonal_order = (0, 1, 1, 12)  # (P, D, Q, s)

    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    results = model.fit(disp=False)

    predictions = results.get_forecast(steps=len(test))
    pred_mean = predictions.predicted_mean
    pred_ci = predictions.conf_int()

    mae = mean_absolute_error(test, pred_mean)
    rmse = np.sqrt(mean_squared_error(test, pred_mean))
    mape = np.mean(np.abs((test - pred_mean) / test)) * 100

    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train, label="Training Data")
    plt.plot(test.index, test, label="Actual Test Data")
    plt.plot(test.index, pred_mean, label="Predictions", color="red")
    plt.fill_between(
        test.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color="red", alpha=0.1
    )
    plt.title("SARIMA Time Series Forecast")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"\nModel Parameters:")
    print(f"SARIMA{order}x{seasonal_order}")
    print(f"\nModel Performance Metrics:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")

    residuals = results.resid
    custom_diagnostics(residuals)

    return results, pred_mean, (mae, rmse, mape)


results, predictions, metrics = train_and_evaluate_sarima(monthly_data)

future_steps = 12
future_forecast = results.get_forecast(steps=future_steps)
future_mean = future_forecast.predicted_mean
future_ci = future_forecast.conf_int()

plt.figure(figsize=(12, 6))
plt.plot(monthly_data.index, monthly_data, label="Historical Data")
plt.plot(future_mean.index, future_mean, label="Future Forecast", color="red")
plt.fill_between(
    future_mean.index,
    future_ci.iloc[:, 0],
    future_ci.iloc[:, 1],
    color="red",
    alpha=0.1,
)
plt.title("SARIMA Future Forecast")
plt.legend()
plt.grid(True)
plt.show()

print("\nForecast for next 12 months:")
for date, value, ci_lower, ci_upper in zip(
    future_mean.index, future_mean.values, future_ci.iloc[:, 0], future_ci.iloc[:, 1]
):
    print(f"{date.strftime('%Y-%m')}: {value:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]")

"""**Rolling Average**"""

# Cyclical Analysis using 12-month rolling average
rolling_avg = monthly_data.rolling(window=12).mean()  # 12-month rolling average

plt.figure(figsize=(10, 6))
plt.plot(monthly_data, label="Original Data", color="blue")
plt.plot(rolling_avg, label="12-Month Rolling Average", color="orange")
plt.title("Cyclical Trend Analysis")
plt.legend()
plt.show()

import joblib


joblib.dump(results, "sarima_model.pkl")
