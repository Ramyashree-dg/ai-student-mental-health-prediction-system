import pandas as pd

# Load dataset
data = pd.read_csv("dataset/mental_health.csv")

print("First 5 rows:\n")
print(data.head())

print("\nColumn Names:\n")
print(data.columns)

print("\nMissing Values:\n")
print(data.isnull().sum())