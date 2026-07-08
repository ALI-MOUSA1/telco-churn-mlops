import pandas as pd

df = pd.read_excel("data/v1/Telco_customer_churn.xlsx")
print("Loaded shape:", df.shape)
df.to_csv("data/v1/telco_churn_raw.csv", index=False)
print("Saved CSV successfully.")