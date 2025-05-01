import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np          #  <-- add this line

# -------------------------------------------------
# 1) Hard-code the data
# -------------------------------------------------
data = {
    "Year": [2020, 2021, 2022, 2023, 2024],
    # Actual Sales-tax receipts for Sugar Land (in millions)
    "Sales_Tax": [109.03, 101.64, 119.84, 155.65, 141.38],
    # Topic-5 “Fiscal & Technology” prevalence scores
    "Fiscal_Technology_Topic5": [0.10, 0.12, 0.15, 0.20, 0.18],
}

df = pd.DataFrame(data)

# -------------------------------------------------
# 2) Calculate Correlation
# -------------------------------------------------
topic = df["Fiscal_Technology_Topic5"]
sales = df["Sales_Tax"]

r, p = pearsonr(topic, sales)
print(f"Correlation coefficient (r) : {r:.4f}")
print(f"P-value                     : {p:.4e}\n")
print(df)

# -------------------------------------------------
# 3) Create Scatter Plot
# -------------------------------------------------
plt.figure(figsize=(8,6))
plt.scatter(topic, sales)
m, b = np.polyfit(topic, sales, 1)
plt.plot(topic, m*topic + b, linestyle="--")
plt.title(f"Topic-5 Prevalence vs. Sales-Tax Revenue\n(r = {r:.2f})")
plt.xlabel("Topic-5 Prevalence")
plt.ylabel("Sales-Tax Revenue (Millions $)")
plt.grid(True)
plt.savefig("correlation_scatterplot.png")
plt.show()
