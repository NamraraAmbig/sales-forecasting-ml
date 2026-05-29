import streamlit as st
import pandas as pd

st.title("Sales Forecasting Dashboard")

st.write("ML Sales Forecasting Project")

df = pd.read_csv("data/cleaned_sales.csv")

st.dataframe(df.head())