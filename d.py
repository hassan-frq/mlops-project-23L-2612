import os
from sklearn.datasets import fetch_california_housing

os.makedirs("data", exist_ok=True)
df = fetch_california_housing(as_frame=True).frame
df.to_csv("data/dataset.csv", index=False)
print("Saved data/dataset.csv", df.shape)