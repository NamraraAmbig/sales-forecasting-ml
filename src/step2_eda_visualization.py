import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import os

sns.set_theme(style="whitegrid")
os.makedirs("outputs", exist_ok=True)
PALETTE = ["#2E86AB","#A23B72","#F18F01","#C73E1D","#44BBA4",
           "#E94F37","#393E41","#F5A623","#7B4F7F","#3B1F2B"]

def load(path="data/cleaned_sales.csv"):
    return pd.read_csv(path, parse_dates=["Order Date","Ship Date"])

def plot_sales_trend(df):
    monthly = df.set_index("Order Date").resample("ME")["Sales"].sum().reset_index()
    monthly.columns = ["Month","Sales"]
    fig, ax = plt.subplots(figsize=(14,5))
    ax.plot(monthly["Month"], monthly["Sales"], color="#2E86AB", linewidth=2, marker="o", markersize=3)
    ax.fill_between(monthly["Month"], monthly["Sales"], alpha=0.12, color="#2E86AB")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f"${x/1000:.0f}K"))
    ax.set_title("Monthly Sales Trend", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig("outputs/01_monthly_sales_trend.png"); plt.close()
    print("[SAVED] outputs/01_monthly_sales_trend.png")

    df["Year"] = df["Order Date"].dt.year
    yearly = df.groupby("Year")["Sales"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(7,4))
    bars = ax.bar(yearly["Year"].astype(str), yearly["Sales"], color=PALETTE[:len(yearly)], width=0.55)
    for bar, val in zip(bars, yearly["Sales"]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01, f"${val/1000:.0f}K", ha="center", fontsize=10, fontweight="bold")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f"${x/1000:.0f}K"))
    ax.set_title("Yearly Sales Comparison", fontweight="bold")
    plt.tight_layout(); plt.savefig("outputs/02_yearly_sales.png"); plt.close()
    print("[SAVED] outputs/02_yearly_sales.png")

def plot_category_performance(df):
    cat = df.groupby("Category")[["Sales","Profit"]].sum().sort_values("Sales",ascending=False)
    fig, axes = plt.subplots(1,2,figsize=(13,5))
    cat["Sales"].plot(kind="barh", ax=axes[0], color=PALETTE[:3])
    axes[0].xaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f"${x/1000:.0f}K"))
    axes[0].set_title("Sales by Category", fontweight="bold")
    cat["Profit"].plot(kind="barh", ax=axes[1], color=PALETTE[3:6])
    axes[1].xaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f"${x/1000:.0f}K"))
    axes[1].set_title("Profit by Category", fontweight="bold")
    plt.tight_layout(); plt.savefig("outputs/03_category_performance.png"); plt.close()
    print("[SAVED] outputs/03_category_performance.png")

    sub = df.groupby("Sub-Category")["Sales"].sum().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10,5))
    sub.plot(kind="bar", ax=ax, color=PALETTE)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f"${x/1000:.0f}K"))
    ax.set_title("Top 10 Sub-Categories by Sales", fontweight="bold")
    plt.xticks(rotation=35, ha="right"); plt.tight_layout()
    plt.savefig("outputs/04_top_subcategories.png"); plt.close()
    print("[SAVED] outputs/04_top_subcategories.png")

def plot_region_analysis(df):
    region = df.groupby("Region")[["Sales","Profit"]].sum().sort_values("Sales",ascending=False)
    fig, axes = plt.subplots(1,2,figsize=(13,5))
    axes[0].pie(region["Sales"], labels=region.index, autopct="%1.1f%%", startangle=140, colors=PALETTE[:4])
    axes[0].set_title("Sales by Region", fontweight="bold")
    x = np.arange(len(region)); w = 0.38
    axes[1].bar(x-w/2, region["Sales"], width=w, label="Sales", color=PALETTE[0])
    axes[1].bar(x+w/2, region["Profit"], width=w, label="Profit", color=PALETTE[1])
    axes[1].set_xticks(x); axes[1].set_xticklabels(region.index)
    axes[1].yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f"${x/1000:.0f}K"))
    axes[1].set_title("Sales vs Profit by Region", fontweight="bold"); axes[1].legend()
    plt.tight_layout(); plt.savefig("outputs/05_region_analysis.png"); plt.close()
    print("[SAVED] outputs/05_region_analysis.png")

def plot_profit_vs_sales(df):
    fig, ax = plt.subplots(figsize=(9,6))
    sc = ax.scatter(df["Sales"], df["Profit"], c=df["Discount"], cmap="RdYlGn_r", alpha=0.45, s=12)
    plt.colorbar(sc, ax=ax, label="Discount Rate")
    ax.axhline(0, color="red", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_title("Profit vs Sales (color = Discount)", fontweight="bold")
    ax.set_xlabel("Sales ($)"); ax.set_ylabel("Profit ($)")
    plt.tight_layout(); plt.savefig("outputs/06_profit_vs_sales.png"); plt.close()
    print("[SAVED] outputs/06_profit_vs_sales.png")

def plot_seasonal_patterns(df):
    df["Month"] = df["Order Date"].dt.month
    df["Year"]  = df["Order Date"].dt.year
    df["MonthName"] = df["Order Date"].dt.strftime("%b")
    month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly_avg = df.groupby("MonthName")["Sales"].mean().reindex(month_order)
    fig, ax = plt.subplots(figsize=(12,4))
    ax.bar(monthly_avg.index, monthly_avg.values, color=PALETTE[:12])
    ax.set_title("Average Sales by Month (Seasonal Pattern)", fontweight="bold")
    plt.tight_layout(); plt.savefig("outputs/07_seasonal_pattern.png"); plt.close()
    print("[SAVED] outputs/07_seasonal_pattern.png")

    pivot = df.pivot_table(values="Sales", index="Year", columns="MonthName", aggfunc="sum").reindex(columns=month_order)
    fig, ax = plt.subplots(figsize=(13,4))
    sns.heatmap(pivot, ax=ax, annot=True, fmt=".0f", cmap="YlOrRd", linewidths=0.4)
    ax.set_title("Sales Heatmap: Year x Month", fontweight="bold")
    plt.tight_layout(); plt.savefig("outputs/08_sales_heatmap.png"); plt.close()
    print("[SAVED] outputs/08_sales_heatmap.png")

def plot_customer_behavior(df):
    seg = df.groupby("Segment")[["Sales","Profit"]].sum()
    fig, axes = plt.subplots(1,2,figsize=(12,5))
    axes[0].pie(seg["Sales"], labels=seg.index, autopct="%1.1f%%", startangle=90, colors=PALETTE[:3])
    axes[0].set_title("Sales by Customer Segment", fontweight="bold")
    ship = df.groupby("Ship Mode")["Sales"].sum().sort_values(ascending=False)
    ship.plot(kind="barh", ax=axes[1], color=PALETTE[:4])
    axes[1].xaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f"${x/1000:.0f}K"))
    axes[1].set_title("Sales by Shipping Mode", fontweight="bold")
    plt.tight_layout(); plt.savefig("outputs/09_customer_behavior.png"); plt.close()
    print("[SAVED] outputs/09_customer_behavior.png")

def run_eda(path="data/cleaned_sales.csv"):
    df = load(path)
    plot_sales_trend(df)
    plot_category_performance(df)
    plot_region_analysis(df)
    plot_profit_vs_sales(df)
    plot_seasonal_patterns(df)
    plot_customer_behavior(df)
    print("[DONE] All EDA charts saved.")
    return df

if __name__ == "__main__":
    run_eda()