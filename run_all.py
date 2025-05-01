#!/usr/bin/env python3
"""
run_all.py

Monolithic workflow to reproduce:
  • Data prep (PDF → CSV, CPI adjustment, imputation)
  • Topic-5 prevalence extraction via LDA
  • Pearson correlation & scatterplot
  • Bayesian-ridge budget forecast & line chart
  • Output forecast table

Assumes this directory structure:
  ├── data/
  │   ├── raw/
  │   │   ├── FY2020-24_ACFR.pdf
  │   │   ├── budget-in-brief-FY2024.pdf
  │   │   ├── sales_tax.csv         # from Texas API
  │   │   └── council_agendas/      # HTML transcripts
  │   └── processed/                # (will be created)
  ├── figures/                      # (will be created)
  └── tables/                       # (will be created)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# ─── Directory setup ──────────────────────────────────────
RAW_DIR       = os.path.join("data", "raw")
PROC_DIR      = os.path.join("data", "processed")
FIG_DIR       = "figures"
TBL_DIR       = "tables"
for d in (PROC_DIR, FIG_DIR, TBL_DIR):
    os.makedirs(d, exist_ok=True)

# ─── 1) Prepare the tabular data ───────────────────────────
def prepare_data():
    # 1a) Extract budgets (assume Tabula has dumped to CSV already)
    #    If you have to re-run from PDF, you could call tabula.read_pdf here.
    budgets = pd.read_csv(os.path.join(RAW_DIR, "sugarland_budgets_2020_24.csv"))
    # 1b) Adjust to 2024 dollars using CPI (hard-coded index factor for example)
    #    (In practice, pull from BLS series CUURA316SA0)
    cpi_2024 = 1.00  # placeholder
    budgets["Total_Budget_2024$M"] = budgets["Total_Budget_$M"] * cpi_2024
    # 1c) Load sales tax
    sales = pd.read_csv(os.path.join(RAW_DIR, "sales_tax.csv"))
    #    Impute Aug–Sep 2024 if missing
    sales["net_payment"].fillna(
        sales.loc[sales.fiscal_year < 2024, "net_payment"].mean(),
        inplace=True
    )
    # 1d) Merge & save
    df = budgets.merge(
        sales[["fiscal_year", "net_payment"]],
        left_on="Fiscal_Year",
        right_on="fiscal_year",
        how="inner"
    )
    df.rename(columns={"net_payment":"Sales_Tax_$M"}, inplace=True)
    df.to_csv(os.path.join(PROC_DIR, "master.csv"), index=False)
    print("→ prepared data/master.csv")

# ─── 2) Extract Topic-5 prevalence ─────────────────────────
def extract_topic5():
    # Load all agendas
    text_data = []
    years = []
    for fname in os.listdir(os.path.join(RAW_DIR, "council_agendas")):
        if not fname.endswith(".html"): continue
        year = int(fname.split("_")[0])
        raw = open(os.path.join(RAW_DIR, "council_agendas", fname)).read()
        # minimal cleaning
        txt = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in raw)
        text_data.append(txt)
        years.append(year)
    # Vectorize + LDA
    vec = CountVectorizer(max_features=2000, stop_words="english")
    X = vec.fit_transform(text_data)
    lda = LatentDirichletAllocation(n_components=8, random_state=0)
    W = lda.fit_transform(X)
    # Identify Topic–5 by index (dominant keywords: cloud, dashboard, ai, billing, budget)
    topic5 = W[:, 4]  # adjust if LDA component labels differ
    df_t = pd.DataFrame({"Fiscal_Year": years, "Topic5_Perv": topic5})
    df_t = df_t.groupby("Fiscal_Year").mean().reset_index()
    df_t.to_csv(os.path.join(PROC_DIR, "topic5.csv"), index=False)
    print("→ prepared data/topic5.csv")

# ─── 3) Correlation & scatterplot ───────────────────────────
def run_correlation():
    df = pd.read_csv(os.path.join(PROC_DIR, "master.csv"))
    t5 = pd.read_csv(os.path.join(PROC_DIR, "topic5.csv"))
    df = df.merge(t5, on="Fiscal_Year")
    r, p = pearsonr(df["Topic5_Perv"], df["Sales_Tax_$M"])
    print(f"Pearson r={r:.2f}, p={p:.3f}")
    plt.figure(figsize=(6,5))
    plt.scatter(df["Topic5_Perv"], df["Sales_Tax_$M"])
    m,b = np.polyfit(df["Topic5_Perv"], df["Sales_Tax_$M"], 1)
    plt.plot(df["Topic5_Perv"], m*df["Topic5_Perv"]+b, linestyle="--")
    plt.xlabel("Topic-5 Prevalence")
    plt.ylabel("Sales-Tax ($M)")
    plt.title(f"r = {r:.2f}, p = {p:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "correlation_scatterplot.png"))
    plt.close()
    print("→ saved figures/correlation_scatterplot.png")

# ─── 4) Bayesian forecast & line chart ──────────────────────
def run_forecast():
    df = pd.read_csv(os.path.join(PROC_DIR, "master.csv"))
    t5 = pd.read_csv(os.path.join(PROC_DIR, "topic5.csv"))
    df = df.merge(t5, on="Fiscal_Year")
    # Train
    X = df[["Fiscal_Year","Sales_Tax_$M","Topic5_Perv"]]
    y = df["Total_Budget_2024$M"]
    model = BayesianRidge()
    model.fit(X, y)
    # In-sample MAPE
    fitted = model.predict(X)
    mape = mean_absolute_percentage_error(y, fitted)*100
    print(f"In-sample MAPE: {mape:.2f}%")
    # Forecast FY25-27
    last_tax   = df["Sales_Tax_$M"].iloc[-1]
    last_topic = df["Topic5_Perv"].iloc[-1]
    future = []
    for i,yr in enumerate((2025,2026,2027), start=1):
        tax    = last_tax * (1.03**i)
        topic  = min(0.25, last_topic + 0.02*i)
        future.append((yr, tax, topic))
    Xf = pd.DataFrame(future, columns=["Fiscal_Year","Sales_Tax_$M","Topic5_Perv"])
    preds = model.predict(Xf)
    df_f = Xf.copy()
    df_f["Forecast_Budget_$M"] = preds.round(2)
    df_f.to_csv(os.path.join(TBL_DIR, "forecast_FY25-27.csv"), index=False)
    print("→ saved tables/forecast_FY25-27.csv")
    # Plot
    plt.figure(figsize=(6,5))
    plt.plot(df["Fiscal_Year"], df["Total_Budget_2024$M"], marker="o", label="Actual")
    plt.plot(df_f["Fiscal_Year"], df_f["Forecast_Budget_$M"], marker="x", linestyle="--", label="Forecast")
    plt.xlabel("Fiscal Year")
    plt.ylabel("Budget ($M)")
    plt.title("Bayesian Ridge Forecast")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "budget_forecast.png"))
    plt.close()
    print("→ saved figures/budget_forecast.png")

# ─── Main ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("1) Preparing data…");   prepare_data()
    print("2) Extracting topic-5…"); extract_topic5()
    print("3) Running correlation…"); run_correlation()
    print("4) Running forecast…");    run_forecast()
    print("✓ All done!")