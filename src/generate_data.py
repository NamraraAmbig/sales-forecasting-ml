import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random, os

np.random.seed(42)
random.seed(42)

N_ROWS = 5000
START_DATE = datetime(2020, 1, 1)
END_DATE   = datetime(2023, 12, 31)

CATEGORIES = {
    "Technology":  ["Phones", "Computers", "Accessories", "Copiers"],
    "Furniture":   ["Chairs", "Tables", "Bookcases", "Furnishings"],
    "Office Supplies": ["Binders", "Paper", "Storage", "Art", "Labels", "Fasteners"],
}
REGIONS = ["West", "East", "Central", "South"]
STATES  = {
    "West":    ["California", "Washington", "Oregon", "Nevada", "Arizona"],
    "East":    ["New York", "Pennsylvania", "Ohio", "Virginia", "Florida"],
    "Central": ["Texas", "Illinois", "Michigan", "Indiana", "Wisconsin"],
    "South":   ["Georgia", "Tennessee", "Alabama", "Louisiana", "Mississippi"],
}
SEGMENTS   = ["Consumer", "Corporate", "Home Office"]
SHIP_MODES = ["First Class", "Second Class", "Standard Class", "Same Day"]

def random_date(start, end):
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))

def base_sales(sub_category):
    bases = {
        "Phones": 400, "Computers": 800, "Accessories": 80, "Copiers": 1200,
        "Chairs": 250, "Tables": 500, "Bookcases": 150, "Furnishings": 60,
        "Binders": 30, "Paper": 15, "Storage": 70, "Art": 12,
        "Labels": 8, "Fasteners": 5,
    }
    return bases.get(sub_category, 50)

def seasonal_multiplier(date):
    multipliers = {1:0.9,2:0.85,3:1.0,4:1.05,5:1.0,6:0.95,
                   7:0.9,8:0.95,9:1.05,10:1.1,11:1.3,12:1.35}
    return multipliers[date.month]

rows = []
for i in range(N_ROWS):
    order_date   = random_date(START_DATE, END_DATE)
    ship_date    = order_date + timedelta(days=random.choice([1,3,5,7]))
    region       = random.choice(REGIONS)
    state        = random.choice(STATES[region])
    segment      = random.choice(SEGMENTS)
    ship_mode    = random.choice(SHIP_MODES)
    category     = random.choice(list(CATEGORIES.keys()))
    sub_category = random.choice(CATEGORIES[category])
    quantity     = random.randint(1, 10)
    base         = base_sales(sub_category)
    price        = base * seasonal_multiplier(order_date) * np.random.uniform(0.8, 1.2)
    sales        = round(price * quantity, 2)
    discount     = round(random.choice([0,0,0,0.1,0.2,0.3,0.4,0.5]), 2)
    profit_rate  = np.random.uniform(0.05, 0.35) * (1 - discount * 0.8)
    profit       = round(sales * profit_rate * (1 - discount), 2)
    rows.append({
        "Order ID": f"US-{order_date.year}-{1000+i:06d}",
        "Order Date": order_date.strftime("%Y-%m-%d"),
        "Ship Date": ship_date.strftime("%Y-%m-%d"),
        "Ship Mode": ship_mode, "Customer ID": f"CUS-{random.randint(1000,9999)}",
        "Segment": segment, "Region": region, "State": state,
        "Category": category, "Sub-Category": sub_category,
        "Sales": sales, "Quantity": quantity, "Discount": discount, "Profit": profit,
    })

df = pd.DataFrame(rows)
for col in ["Ship Mode", "State", "Discount"]:
    mask = np.random.rand(len(df)) < 0.02
    df.loc[mask, col] = np.nan

os.makedirs("data", exist_ok=True)
df.to_csv("data/superstore_sales.csv", index=False)
print(f"Dataset created: {len(df)} rows")