import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


STUDENT_ID = "23L-2612"

def main():
    print(f"[{STUDENT_ID}] Loading dataset...")
    df = pd.read_csv("data/dataset.csv")

    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]

    X = MinMaxScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"[{STUDENT_ID}] Training RandomForest model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print(f"[{STUDENT_ID}] Model R^2 score: {score:.4f}")

    joblib.dump(model, "model/model_23L-2612.pkl")
    print(f"[{STUDENT_ID}] Model saved to model/model_23L-2612.pkl")

if __name__ == "__main__":
    main()