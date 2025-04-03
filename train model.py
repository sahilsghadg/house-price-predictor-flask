# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor # Using RandomForest as the model
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings
import os

# --- Configuration ---
DATA_FILENAME = 'Housing.csv'
PIPELINE_FILENAME = 'house_price_pipeline.joblib'
RANDOM_STATE = 42 # for reproducibility

# Ignore warnings for cleaner output (optional)
warnings.filterwarnings('ignore')

print("--- Starting Model Training Script ---")

# --- 1. Load Data ---
print(f"Loading data from '{DATA_FILENAME}'...")
if not os.path.exists(DATA_FILENAME):
    print(f"Error: Data file '{DATA_FILENAME}' not found in the current directory.")
    print("Please make sure the CSV file is present.")
    exit() # Stop script if data file is missing

try:
    df = pd.read_csv(DATA_FILENAME)
    print("Data loaded successfully.")
    print("Shape of data:", df.shape)
except Exception as e:
    print(f"Error loading or reading the CSV file: {e}")
    exit()

# --- 2. Initial Inspection ---
print("\nData Info:")
df.info()

print("\nMissing Values Check:")
print(df.isnull().sum())
# Optional: Add code here to handle missing values if any were found

# --- 3. Define Features and Target ---
TARGET = 'price'

# Identify feature types
binary_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
multi_cat_features = ['furnishingstatus']
# All remaining columns (excluding target and categorical) are assumed numerical
numerical_features = [col for col in df.columns if col not in [TARGET] + binary_features + multi_cat_features]

print(f"\nTarget variable: {TARGET}")
print(f"Numerical features identified: {numerical_features}")
print(f"Binary features (yes/no): {binary_features}")
print(f"Multi-class categorical features: {multi_cat_features}")

# --- 4. Preprocessing Steps ---

# Create a copy to avoid modifying the original DataFrame directly
df_processed = df.copy()

# Map 'yes'/'no' to 1/0 for binary features
print("\nMapping binary features (yes=1, no=0)...")
for col in binary_features:
    if col in df_processed.columns:
         df_processed[col] = df_processed[col].map({'yes': 1, 'no': 0}).astype(int)
    else:
        print(f"Warning: Binary feature '{col}' not found in DataFrame.")

# Now, treat these mapped binary features as numerical for scaling purposes
numerical_features_including_binary = numerical_features + binary_features
print(f"Features to be scaled: {numerical_features_including_binary}")


# --- Create Preprocessing Pipelines for different column types ---

# Pipeline for numerical features (including mapped binary): Scale them
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Pipeline for the multi-class categorical feature: One-Hot Encode it
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
])

# --- Combine preprocessing steps using ColumnTransformer ---
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features_including_binary),
        ('cat', categorical_transformer, multi_cat_features)
    ],
    remainder='passthrough' # Handle any unexpected columns gracefully, though ideally all are covered
)

# --- 5. Split Data ---
print("\nSplitting data into Training and Testing sets...")
X = df_processed.drop(TARGET, axis=1)
y = df_processed[TARGET]

# Ensure all defined features exist in X before splitting
feature_list_for_pipeline = numerical_features_including_binary + multi_cat_features
missing_cols = [col for col in feature_list_for_pipeline if col not in X.columns]
if missing_cols:
    print(f"Error: The following features defined for preprocessing are missing from the DataFrame: {missing_cols}")
    exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
print(f"Testing data shape: X={X_test.shape}, y={y_test.shape}")

# --- 6. Create the Full ML Pipeline (Preprocessing + Model) ---
print("\nDefining the full Machine Learning Pipeline (Preprocessing + RandomForestRegressor)...")
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1))
])

# --- 7. Train the Model ---
print("\nTraining the RandomForest model pipeline...")
try:
    model_pipeline.fit(X_train, y_train)
    print("Model Training Complete.")
except Exception as e:
    print(f"Error during model training: {e}")
    exit()

# --- 8. Evaluate Model (Optional but Recommended) ---
print("\nEvaluating model performance on the Test set...")
try:
    y_pred = model_pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R²): {r2:.4f}")
except Exception as e:
    print(f"Error during model evaluation: {e}")

# --- 9. Save the Trained Model Pipeline ---
print(f"\nSaving the trained pipeline to '{PIPELINE_FILENAME}'...")
try:
    joblib.dump(model_pipeline, PIPELINE_FILENAME)
    print(f"Pipeline successfully saved as '{PIPELINE_FILENAME}'")
except Exception as e:
    print(f"Error saving the pipeline: {e}")

print("\n--- Model Training Script Finished ---")