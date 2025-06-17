# House Price Predictor using XGBoost

# Step 1: Import Libraries
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Step 2: Load Data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Step 3: Combine for preprocessing
train_df["source"] = "train"
test_df["source"] = "test"
test_df["SalePrice"] = np.nan
data = pd.concat([train_df, test_df], axis=0)

# Step 4: Drop columns with too many missing values and irrelevant columns
data.drop(columns=["Alley", "PoolQC", "Fence", "MiscFeature", "Id"], inplace=True)

# Step 5: Handle missing values
for col in data.columns:
    if data[col].isnull().sum() > 0:
        if data[col].dtype == "object":
            data[col].fillna(data[col].mode()[0], inplace=True)
        else:
            data[col].fillna(data[col].median(), inplace=True)

# Step 6: Encode categorical columns
data = pd.get_dummies(data, drop_first=True)

# Step 7: Split data back into train and test
data["source_train"] = data["source_train"] = (data["source_train"] if "source_train" in data.columns else (data["source"] == "train")).astype(int)
train_data = data[data["source_train"] == 1].drop(["source", "source_train"], axis=1)
test_data = data[data["source_train"] == 0].drop(["SalePrice", "source", "source_train"], axis=1)

X = train_data.drop("SalePrice", axis=1)
y = train_data["SalePrice"]

# Step 8: Train/test split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 9: Train XGBoost model
model = XGBRegressor()
model.fit(X_train, y_train)

# Step 10: Predict and evaluate
y_pred = model.predict(X_valid)
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
print("Validation RMSE:", rmse)

# Step 11: Predict on test data
test_preds = model.predict(test_data)

# Step 12: Prepare submission file
submission = pd.read_csv("sample_submission.csv")
submission["SalePrice"] = test_preds
submission.to_csv("submission.csv", index=False)

