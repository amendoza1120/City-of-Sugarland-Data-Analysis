import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_percentage_error

# -------------------------------------------------
# 1) Data (same five years you used)
# -------------------------------------------------
data = {
    "Year": [2020, 2021, 2022, 2023, 2024],
    "Sales_Tax": [109.03, 101.64, 119.84, 155.65, 141.38],          # M$
    "Topic5":     [0.10,   0.12,   0.15,   0.20,   0.18],           # prevalence
    "Total_Budget": [272.568, 254.099, 299.603, 389.118, 353.442],  # M$
}
df = pd.DataFrame(data)

# -------------------------------------------------
# 2) Build feature matrix X and target y
# -------------------------------------------------
X = df[["Year", "Sales_Tax", "Topic5"]]
y = df["Total_Budget"]

# -------------------------------------------------
# 3) Fit Bayesian-regularized regression
# -------------------------------------------------
model = BayesianRidge()
model.fit(X, y)

# In-sample fitted values & MAPE
fitted = model.predict(X)
mape = mean_absolute_percentage_error(y, fitted) * 100
print(f"In-sample MAPE: {mape:.2f}%")

# -------------------------------------------------
# 4) Forecast FY2025-FY2027
# -------------------------------------------------
future_years = [2025, 2026, 2027]

# Assumptions:
#   – Sales-tax grows 3 % per year from the FY2024 base
#   – Topic-5 prevalence continues the 5-year trend (+0.02 each year, capped at 0.25)
last_sales = df.loc[df.index[-1], "Sales_Tax"]
last_topic = df.loc[df.index[-1], "Topic5"]

future_sales = [round(last_sales * (1.03 ** (i+1)), 2) for i in range(3)]
future_topic = [min(0.25, round(last_topic + 0.02*(i+1), 2)) for i in range(3)]

X_future = pd.DataFrame({
    "Year": future_years,
    "Sales_Tax": future_sales,
    "Topic5": future_topic,
})

forecast = model.predict(X_future)

results = pd.DataFrame({
    "Fiscal Year": future_years,
    "Sales_Tax (assumed)": future_sales,
    "Topic5 (assumed)": future_topic,
    "Forecast Budget (M$)": np.round(forecast, 2),
})

print("\n=== Forecast FY2025–FY2027 ===")
print(results.to_string(index=False))

# -------------------------------------------------
# 5) Plot actual vs. forecast
# -------------------------------------------------
plt.figure(figsize=(8,6))
plt.plot(df["Year"], y, marker="o", label="Actual Budget")
plt.plot(results["Fiscal Year"], results["Forecast Budget (M$)"],
         marker="x", linestyle="--", label="Forecast Budget")
plt.xlabel("Fiscal Year")
plt.ylabel("Budget (Millions $)")
plt.title("Bayesian Forecast of Sugar Land Budget")
plt.legend()
plt.grid(True)
plt.savefig("budget_forecast.png")
plt.show()
