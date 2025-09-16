import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
url = "https://raw.githubusercontent.com/ManikantaSanjay/crop_yield_prediction_regression/master/yield_df.csv"
df = pd.read_csv(url)

# Drop index column if exists
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Encode categorical features
le = LabelEncoder()
for col in ["Area", "Item"]:
    df[col] = le.fit_transform(df[col])

# Features and target
X = df.drop("hg/ha_yield", axis=1)
y = df["hg/ha_yield"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    results[name] = {"R²": r2, "RMSE": rmse}
    print(f"{name}: R²={r2:.3f}, RMSE={rmse:.3f}")

# Save the best model
best_model_name = max(results, key=lambda k: results[k]["R²"])
joblib.dump(models[best_model_name], "best_model.pkl")
print(f"\n✅ Saved best model: {best_model_name}")
