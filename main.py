import sys, os
sys.path.insert(0, "src")
os.makedirs("data",    exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("models",  exist_ok=True)

print("=" * 60)
print(" SALES FORECASTING & BUSINESS ANALYTICS SYSTEM")
print("=" * 60)

if not os.path.exists("data/superstore_sales.csv"):
    print("\n[STEP 0] Generating dataset...")
    exec(open("src/generate_data.py").read())
else:
    print("\n[STEP 0] Dataset already exists - skipping.")

print("\n STEP 1: DATA CLEANING")
from step1_data_cleaning import clean_pipeline
clean_pipeline("data/superstore_sales.csv")

print("\n STEP 2 & 3: EDA & VISUALIZATION")
from step2_eda_visualization import run_eda
run_eda("data/cleaned_sales.csv")

print("\n STEP 4,5,6: ML + EVALUATION")
from step3_ml_models import run_ml_pipeline
run_ml_pipeline("data/cleaned_sales.csv")

print("\n DONE! Check the outputs/ folder for all charts.")