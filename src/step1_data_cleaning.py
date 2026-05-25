import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
os.makedirs("outputs", exist_ok=True)

def load_data(path):
    df = pd.read_csv(path)
    print(f"[LOAD] Shape: {df.shape}")
    return df

def handle_missing(df):
    print("\n[MISSING VALUES BEFORE]")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    for col in df.select_dtypes(include="number").columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include="str").columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    print(f"[MISSING VALUES AFTER] {df.isnull().sum().sum()} remaining")
    return df

def remove_duplicates(df):
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"[DUPLICATES] Removed {before - len(df)} rows")
    return df

def convert_dates(df):
    for col in ["Order Date", "Ship Date"]:
        df[col] = pd.to_datetime(df[col])
    print("[DATE] Converted Order Date and Ship Date")
    return df

def handle_outliers(df):
    for col in ["Sales", "Profit"]:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        before = len(df)
        df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]
        print(f"[OUTLIERS] '{col}': kept {len(df)}/{before} rows")
    return df.reset_index(drop=True)

def encode_categories(df):
    le = LabelEncoder()
    for col in ["Segment", "Region", "Category", "Ship Mode"]:
        if col in df.columns:
            df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
    print("[ENCODE] Categorical columns encoded")
    return df

def clean_pipeline(path="data/superstore_sales.csv"):
    df = load_data(path)
    df = handle_missing(df)
    df = remove_duplicates(df)
    df = convert_dates(df)
    df = handle_outliers(df)
    df = encode_categories(df)
    df.to_csv("data/cleaned_sales.csv", index=False)
    print(f"[DONE] Cleaned data saved. Shape={df.shape}")
    return df

if __name__ == "__main__":
    clean_pipeline()