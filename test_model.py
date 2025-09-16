import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

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

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model (avoids missing/corrupted pickle issues in CI)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Save model optionally (for local use)
joblib.dump(model, "best_model.pkl")

# Predictions
y_pred = model.predict(X_test)

# Simple test: R² must be above 0.70
def test_model_performance():
    r2 = r2_score(y_test, y_pred)
    assert r2 > 0.70, f"❌ R² too low: {r2:.3f}"
