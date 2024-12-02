import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import seaborn as sns

warnings.filterwarnings("ignore")

# Read and prepare data
df = pd.read_csv(r"E:\Projects\walmart_analytics\cleaned_data.csv")
df["date"] = pd.to_datetime(df["date"])

# Create monthly aggregation
monthly_data = (
    df.groupby(pd.Grouper(key="date", freq="M"))
    .agg({"total_price": "sum", "quantity": "sum", "unit_price": "mean"})
    .reset_index()
)
monthly_data.set_index("date", inplace=True)

# Add seasonal features
monthly_data["month"] = monthly_data.index.month
monthly_data["quarter"] = monthly_data.index.quarter

# Analyze and handle negative values
print("\nData Analysis:")
print("Minimum value:", monthly_data["total_price"].min())
print("Maximum value:", monthly_data["total_price"].max())
print("Number of negative values:", sum(monthly_data["total_price"] < 0))

# Handle negative values by adding a constant
if monthly_data["total_price"].min() < 0:
    shift_factor = abs(monthly_data["total_price"].min()) + 1
    monthly_data["total_price_adjusted"] = monthly_data["total_price"] + shift_factor
else:
    monthly_data["total_price_adjusted"] = monthly_data["total_price"]

# Plot seasonal patterns
plt.figure(figsize=(15, 6))
sns.boxplot(x="month", y="total_price_adjusted", data=monthly_data)
plt.title("Monthly Sales Patterns")
plt.show()

# Split data into train and test
train_size = int(len(monthly_data) * 0.8)
train = monthly_data["total_price_adjusted"][:train_size]
test = monthly_data["total_price_adjusted"][train_size:]


def optimize_hw_model(train, test, seasonal_periods=12):
    """Optimize Holt-Winters model parameters"""
    best_rmse = float("inf")
    best_model = None
    best_params = None

    # Grid search for parameters
    alphas = [0.1, 0.2, 0.3]
    betas = [0.1, 0.2, 0.3]
    gammas = [0.1, 0.2, 0.3]

    for alpha in alphas:
        for beta in betas:
            for gamma in gammas:
                try:
                    model = ExponentialSmoothing(
                        train,
                        trend="add",
                        seasonal="add",
                        seasonal_periods=seasonal_periods,
                    ).fit(
                        smoothing_level=alpha,
                        smoothing_trend=beta,
                        smoothing_seasonal=gamma,
                    )

                    predictions = model.forecast(len(test))
                    rmse = np.sqrt(mean_squared_error(test, predictions))

                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_model = model
                        best_params = {"alpha": alpha, "beta": beta, "gamma": gamma}
                except:
                    continue

    return best_model, best_params, best_rmse


# Optimize model
print("\nOptimizing Holt-Winters model...")
best_model, best_params, best_rmse = optimize_hw_model(train, test)

# Generate predictions
predictions = best_model.forecast(len(test))
future_forecast = best_model.forecast(12)

# Calculate metrics
mae = mean_absolute_error(test, predictions)
mape = np.mean(np.abs((test - predictions) / test)) * 100

print("\nBest Model Parameters:")
print(f"Alpha (level): {best_params['alpha']}")
print(f"Beta (trend): {best_params['beta']}")
print(f"Gamma (seasonal): {best_params['gamma']}")

print("\nModel Performance Metrics:")
print(f"RMSE: {best_rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MAPE: {mape:.2f}%")

# Plot results
plt.figure(figsize=(15, 7))
plt.plot(
    monthly_data.index,
    monthly_data["total_price_adjusted"],
    label="Actual",
    color="blue",
)
plt.plot(test.index, predictions, label="Forecast", color="red", linestyle="--")
plt.title("Holt-Winters Forecast vs Actual")
plt.legend()
plt.xticks(rotation=45)
plt.show()

# Plot future forecast
future_index = pd.date_range(start=monthly_data.index[-1], periods=13, freq="M")[1:]

plt.figure(figsize=(15, 7))
plt.plot(
    monthly_data.index,
    monthly_data["total_price_adjusted"],
    label="Historical Data",
    color="blue",
)
plt.plot(
    future_index, future_forecast, label="Future Forecast", color="red", linestyle="--"
)
plt.title("12-Month Sales Forecast")
plt.legend()
plt.xticks(rotation=45)
plt.show()

# Calculate prediction intervals
residuals = best_model.resid
residual_std = np.std(residuals)
confidence_interval = 1.96 * residual_std  # 95% confidence interval

# Print forecasts with confidence intervals
print("\nForecast Values for Next 12 Months:")
print("Month\t\tForecast\tLower CI\tUpper CI")
print("-" * 60)
for date, forecast in zip(future_index, future_forecast):
    lower_ci = forecast - confidence_interval
    upper_ci = forecast + confidence_interval

    # Adjust back if we shifted the values
    if "shift_factor" in locals():
        forecast = forecast - shift_factor
        lower_ci = lower_ci - shift_factor
        upper_ci = upper_ci - shift_factor

    print(f"{date.strftime('%Y-%m')}\t{forecast:.2f}\t{lower_ci:.2f}\t{upper_ci:.2f}")

# Additional analysis
monthly_growth = monthly_data["total_price"].pct_change()
print("\nMonthly Growth Statistics:")
print(monthly_growth.describe())

# Seasonal strength
seasonal_strength = monthly_data.groupby("month")["total_price"].mean()
print("\nSeasonal Pattern Strength:")
print(seasonal_strength)
