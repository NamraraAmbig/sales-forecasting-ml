import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import warnings, os, pickle
warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_theme(style="whitegrid")
os.makedirs("outputs", exist_ok=True)
os.makedirs("models",  exist_ok=True)

FEATURE_COLS = ["Month","Year","Quarter","MonthSin","MonthCos",
                "IsHolidaySeason","Sales_lag1","Sales_lag2",
                "Sales_lag3","Sales_lag6","Rolling3M","Rolling6M"]

def build_features(df):
    df = df.copy()
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["YearMonth"]  = df["Order Date"].dt.to_period("M")
    monthly = df.groupby("YearMonth")["Sales"].sum().reset_index().sort_values("YearMonth")
    monthly["YearMonth_dt"] = monthly["YearMonth"].dt.to_timestamp()
    monthly["Month"]   = monthly["YearMonth_dt"].dt.month
    monthly["Year"]    = monthly["YearMonth_dt"].dt.year
    monthly["Quarter"] = monthly["YearMonth_dt"].dt.quarter
    monthly["MonthSin"] = np.sin(2*np.pi*monthly["Month"]/12)
    monthly["MonthCos"] = np.cos(2*np.pi*monthly["Month"]/12)
    monthly["IsHolidaySeason"] = monthly["Month"].isin([10,11,12]).astype(int)
    for lag in [1,2,3,6]:
        monthly[f"Sales_lag{lag}"] = monthly["Sales"].shift(lag)
    monthly["Rolling3M"] = monthly["Sales"].shift(1).rolling(3).mean()
    monthly["Rolling6M"] = monthly["Sales"].shift(1).rolling(6).mean()
    monthly["SalesGrowth"] = monthly["Sales"].pct_change().replace([np.inf,-np.inf],0)
    monthly.dropna(inplace=True)
    monthly.reset_index(drop=True, inplace=True)
    print(f"[FEATURES] Matrix shape: {monthly.shape}")
    return monthly

def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"\n Model: {name}")
    print(f" MAE  : ${mae:,.2f}")
    print(f" RMSE : ${rmse:,.2f}")
    print(f" R2   : {r2:.4f}")
    return {"Model":name,"MAE":mae,"RMSE":rmse,"R2":r2}

def plot_predictions(monthly, split_idx, preds_dict):
    fig, ax = plt.subplots(figsize=(14,5))
    dates = monthly["YearMonth_dt"]
    ax.plot(dates, monthly["Sales"], label="Actual", color="#2E86AB", linewidth=2)
    colors = ["#F18F01","#C73E1D","#44BBA4"]
    styles = ["--","-.",":" ]
    for (name,pred),c,ls in zip(preds_dict.items(),colors,styles):
        ax.plot(dates.iloc[split_idx:], pred, label=f"{name}", color=c, linewidth=2, linestyle=ls)
    ax.axvline(dates.iloc[split_idx], color="gray", linestyle="--", alpha=0.5)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f"${x/1000:.0f}K"))
    ax.set_title("Sales Forecast: Actual vs Predicted", fontweight="bold")
    ax.legend(); plt.tight_layout()
    plt.savefig("outputs/10_forecast_comparison.png"); plt.close()
    print("[SAVED] outputs/10_forecast_comparison.png")

def plot_model_comparison(results):
    df_res = pd.DataFrame(results)
    fig, axes = plt.subplots(1,3,figsize=(13,4))
    for ax,(metric,note),c in zip(axes,[("MAE","lower"),("RMSE","lower"),("R2","higher")],
                                       ["#C73E1D","#F18F01","#44BBA4"]):
        ax.bar(df_res["Model"], df_res[metric], color=c, alpha=0.85)
        ax.set_title(f"{metric} ({note} is better)", fontweight="bold")
        for bar,val in zip(ax.patches, df_res[metric]):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01,
                    f"{val:,.0f}" if metric!="R2" else f"{val:.3f}",
                    ha="center", fontsize=9)
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    plt.suptitle("Model Evaluation Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.savefig("outputs/11_model_comparison.png",bbox_inches="tight"); plt.close()
    print("[SAVED] outputs/11_model_comparison.png")

def plot_feature_importance(rf_model):
    imp = pd.Series(rf_model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9,6))
    imp.plot(kind="barh", ax=ax, color="#2E86AB", alpha=0.85)
    ax.set_title("Random Forest - Feature Importance", fontweight="bold")
    plt.tight_layout(); plt.savefig("outputs/12_feature_importance.png"); plt.close()
    print("[SAVED] outputs/12_feature_importance.png")

def forecast_future(monthly, best_model, n_months=6):
    history = monthly["Sales"].values.tolist()
    last_date = monthly["YearMonth_dt"].iloc[-1]
    future_rows = []
    for i in range(1, n_months+1):
        next_date = last_date + pd.DateOffset(months=i)
        month = next_date.month
        row = {
            "Month":month,"Year":next_date.year,"Quarter":(month-1)//3+1,
            "MonthSin":np.sin(2*np.pi*month/12),"MonthCos":np.cos(2*np.pi*month/12),
            "IsHolidaySeason":int(month in [10,11,12]),
            "Sales_lag1":history[-1],"Sales_lag2":history[-2] if len(history)>=2 else history[-1],
            "Sales_lag3":history[-3] if len(history)>=3 else history[-1],
            "Sales_lag6":history[-6] if len(history)>=6 else history[-1],
            "Rolling3M":np.mean(history[-3:]),"Rolling6M":np.mean(history[-6:]),
        }
        pred = max(best_model.predict(pd.DataFrame([row])[FEATURE_COLS])[0], 0)
        history.append(pred)
        future_rows.append({"Date":next_date,"Forecasted_Sales":pred})
    future_df = pd.DataFrame(future_rows)
    fig, ax = plt.subplots(figsize=(13,5))
    ax.plot(monthly["YearMonth_dt"], monthly["Sales"], label="Historical", color="#2E86AB", linewidth=2)
    ax.plot(future_df["Date"], future_df["Forecasted_Sales"], label="Forecast",
            color="#F18F01", linewidth=2.5, linestyle="--", marker="o", markersize=7)
    ax.fill_between(future_df["Date"], future_df["Forecasted_Sales"]*0.9,
                    future_df["Forecasted_Sales"]*1.1, alpha=0.15, color="#F18F01")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f"${x/1000:.0f}K"))
    ax.set_title(f"Sales Forecast - Next {n_months} Months", fontweight="bold")
    ax.legend(); plt.tight_layout()
    plt.savefig("outputs/13_future_forecast.png"); plt.close()
    print("[SAVED] outputs/13_future_forecast.png")
    print("\n[FUTURE FORECAST]")
    future_df["Date"] = future_df["Date"].dt.strftime("%b %Y")
    future_df["Forecasted_Sales"] = future_df["Forecasted_Sales"].map("${:,.0f}".format)
    print(future_df.to_string(index=False))
    return future_df

def run_ml_pipeline(path="data/cleaned_sales.csv"):
    df = pd.read_csv(path)
    monthly = build_features(df)
    X = monthly[FEATURE_COLS]; y = monthly["Sales"]
    split_idx = int(len(monthly)*0.8)
    X_train,X_test = X.iloc[:split_idx],X.iloc[split_idx:]
    y_train,y_test = y.iloc[:split_idx],y.iloc[split_idx:]

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    results = {}
    preds_dict = {}

    lr = LinearRegression()
    lr.fit(X_tr_sc, y_train)
    lr_pred = lr.predict(X_te_sc)
    results["Linear Regression"] = evaluate("Linear Regression", y_test, lr_pred)
    preds_dict["Linear Regression"] = lr_pred
    pickle.dump(lr, open("models/linear_regression.pkl","wb"))

    rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results["Random Forest"] = evaluate("Random Forest", y_test, rf_pred)
    preds_dict["Random Forest"] = rf_pred
    pickle.dump(rf, open("models/random_forest.pkl","wb"))
    plot_feature_importance(rf)

    plot_predictions(monthly, split_idx, preds_dict)
    plot_model_comparison(list(results.values()))

    best_name = max(results, key=lambda k: results[k]["R2"])
    print(f"\n[BEST MODEL] {best_name}  R2={results[best_name]['R2']:.4f}")
    best_model = pickle.load(open(f"models/{best_name.replace(' ','_').lower()}.pkl","rb"))
    future = forecast_future(monthly, best_model, n_months=6)
    pd.DataFrame(list(results.values())).round(2).to_csv("outputs/model_results.csv", index=False)
    print("[DONE] All models complete.")
    return list(results.values()), future

if __name__ == "__main__":
    run_ml_pipeline()