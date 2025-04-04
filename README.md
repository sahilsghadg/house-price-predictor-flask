# House Price Predictor (Flask)

A web application built with Flask and Scikit-learn to predict house prices based on various features. This project was created as part of the Advanced Analytics course activities.

**Live Demo:** [https://house-price-predictor-flask.onrender.com](https://house-price-predictor-flask.onrender.com)

*(Note: The free tier on Render may cause the app to sleep after inactivity. The first load might take 15-30 seconds.)*

## How the Application Works

The application provides a simple web interface for users to input house characteristics and receive an estimated price prediction.

**User Interface (`templates/index.html`):**

1.  Users access the application via the public URL.
2.  An HTML form is presented, allowing input for features like:
    *   Area (sqft)
    *   Number of Bedrooms, Bathrooms, Stories, Parking spaces
    *   Amenities (Main Road access, Guestroom, Basement, Hot Water, AC) - selected via Yes/No dropdowns.
    *   Location Preference (Preferred Area) - selected via Yes/No dropdown.
    *   Furnishing Status (Furnished, Semi-furnished, Unfurnished) - selected via dropdown.
3.  Upon submitting the form ("Predict Price" button), the data is sent to the Flask backend.

**Backend Logic (`app.py`):**

1.  **Route Handling:** Flask listens for requests.
    *   `GET /`: Renders the initial `index.html` form.
    *   `POST /predict`: Handles the submitted form data.
2.  **Data Reception:** Extracts the user's input values from the form data. Basic validation ensures all fields are present.
3.  **Input Preprocessing (App-side):**
    *   Crucially, the binary 'yes'/'no' string values received from the form for amenities and preference are mapped to integers (1 for 'yes', 0 for 'no'). This is necessary because the prediction pipeline expects numerical input for these features *before* its own internal preprocessing (like scaling).
4.  **DataFrame Creation:** The processed input values are organized into a Pandas DataFrame. The structure and column names of this DataFrame match the original dataset structure used during training (e.g., 'mainroad' column now contains 1 or 0).
5.  **Load Pipeline:** The pre-trained Scikit-learn pipeline, saved as `house_price_pipeline.joblib`, is loaded using `joblib`. This pipeline object contains *all* the necessary steps: preprocessing (scaling, encoding) and the trained RandomForest model.
6.  **Prediction:** The `.predict()` method of the loaded pipeline is called on the input DataFrame. The pipeline automatically applies the appropriate scaling (StandardScaler) and encoding (OneHotEncoder) to the input data before feeding it to the RandomForestRegressor model to generate a prediction.
7.  **Response:** The raw numerical prediction is formatted into a user-friendly currency string (e.g., "₹ 4,500,000.00").
8.  **Render Result:** The `index.html` template is rendered again, passing the prediction text (or an error message if something went wrong) back to the user for display. The form retains the user's previous input values for convenience.

## Model Training and Preprocessing (`train_model.py`)

The prediction model was trained offline using the `Housing.csv` dataset. The process involved several key steps encapsulated within a Scikit-learn Pipeline:

1.  **Data Loading:** The `Housing.csv` dataset is loaded into a Pandas DataFrame.
2.  **Feature Identification:** Columns are categorized into:
    *   Numerical: `area`, `bedrooms`, `bathrooms`, `stories`, `parking`
    *   Binary Categorical ('yes'/'no'): `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`
    *   Multi-Class Categorical: `furnishingstatus`
    *   Target: `price`
3.  **Initial Preprocessing:** Binary 'yes'/'no' features are mapped to integers 1 and 0, respectively.
4.  **Pipeline Definition:** A Scikit-learn `Pipeline` is constructed containing:
    *   **Preprocessor (`ColumnTransformer`):** This applies different transformations to different columns simultaneously:
        *   `StandardScaler`: Applied to all numerical features *and* the already mapped binary (0/1) features. This scales features to have zero mean and unit variance.
        *   `OneHotEncoder`: Applied to the `furnishingstatus` column. It converts the categorical text ('furnished', 'semi-furnished', 'unfurnished') into numerical columns. `drop='first'` is used to avoid multicollinearity.
    *   **Regressor (`RandomForestRegressor`):** An ensemble learning model chosen for this regression task (`n_estimators=100`, `random_state=42`).
5.  **Data Splitting:** The dataset is split into training (80%) and testing (20%) sets.
6.  **Training:** The *entire pipeline* (preprocessor + regressor) is trained using the `.fit()` method on the training data (`X_train`, `y_train`). This ensures the scaler and encoder are fitted *only* on the training data, preventing data leakage.
7.  **Evaluation (Informational):** The trained pipeline's performance is evaluated on the unseen test set using metrics like Root Mean Squared Error (RMSE) and R-squared (R²) to gauge its accuracy.
8.  **Saving:** The fully trained `Pipeline` object (containing the fitted preprocessor and the fitted model) is saved into a single file, `house_price_pipeline.joblib`, using `joblib.dump()`. This file is then loaded by the Flask application (`app.py`) to make predictions on new data.

## Technology Stack

*   **Backend:** Python, Flask
*   **ML/Data:** Scikit-learn, Pandas, NumPy, Joblib
*   **Web Server (Deployment):** Gunicorn
*   **Frontend:** HTML, CSS (via Flask Templates/Jinja2)
*   **Deployment:** Render.com
*   **Version Control:** Git, GitHub

## Setup and Running Locally

1.  **Clone:** `git clone https://github.com/sahilsghadg/house-price-predictor-flask.git`
2.  **Navigate:** `cd house-price-predictor-flask`
3.  **Create Environment:** `python -m venv venv` (or use your preferred method)
4.  **Activate:**
    *   Windows: `.\venv\Scripts\activate`
    *   macOS/Linux: `source venv/bin/activate`
5.  **Install Dependencies:** `pip install -r requirements.txt`
6.  **Ensure Model Exists:** Make sure `house_price_pipeline.joblib` is present. If not, run the training script (requires `Housing.csv`): `python train_model.py`
7.  **Run App:** `python app.py`
8.  **Access:** Open your browser to `http://127.0.0.1:5001` (or the URL shown in the terminal).

## Code Files

Below are the final versions of the core code files used in this project.

---

### `train_model.py`

```python
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

app.py
# app.py (Updated)
import os
import warnings
import pandas as pd
import joblib
from flask import Flask, request, render_template, flash
import traceback # For detailed error logging if needed

# --- Configuration & Setup ---
warnings.filterwarnings('ignore')  # Suppress warnings

app = Flask(__name__)  # Initialize Flask app
# IMPORTANT: Use a strong, environment-variable-based secret key in production
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'a_very_secure_default_dev_key_!@#$%^')

PIPELINE_FILENAME = 'house_price_pipeline.joblib'

# --- Load Model Pipeline ---
pipeline = None
pipeline_load_error = None # Store potential loading error message
if os.path.exists(PIPELINE_FILENAME):
    try:
        pipeline = joblib.load(PIPELINE_FILENAME)
        print(f"Model pipeline '{PIPELINE_FILENAME}' loaded successfully.")
    except Exception as e:
        pipeline_load_error = f"Error loading model pipeline '{PIPELINE_FILENAME}': {e}"
        print(pipeline_load_error)
        # Keep pipeline as None
else:
    pipeline_load_error = f"Error: Model pipeline file '{PIPELINE_FILENAME}' not found. Please run training script."
    print(pipeline_load_error)
    # Keep pipeline as None

# --- Flask Routes ---

@app.route('/')
def home():
    """Renders the initial form page."""
    if pipeline_load_error:
        # Show error only on first load if pipeline failed
        flash(pipeline_load_error, "error")
    # Pass the request object itself, which will be empty on initial GET
    return render_template('index.html', request=request)


@app.route('/predict', methods=['POST'])
def predict():
    """Handles form submission, makes prediction, and renders the result."""
    # Always check if pipeline is loaded before attempting prediction
    if pipeline is None:
        flash("Prediction cannot be made because the model pipeline is not loaded.", "error")
        # Pass original form data back to template to retain user input
        return render_template('index.html', request=request.form)

    prediction_result_text = None
    form_data = request.form # Get the form data once

    try:
        # 1. Get raw data from the POST request form
        raw_input_values = {
            'area': form_data.get('area', type=int),
            'bedrooms': form_data.get('bedrooms', type=int),
            'bathrooms': form_data.get('bathrooms', type=int),
            'stories': form_data.get('stories', type=int),
            'mainroad': form_data.get('mainroad', type=str),
            'guestroom': form_data.get('guestroom', type=str),
            'basement': form_data.get('basement', type=str),
            'hotwaterheating': form_data.get('hotwaterheating', type=str),
            'airconditioning': form_data.get('airconditioning', type=str),
            'parking': form_data.get('parking', type=int),
            'prefarea': form_data.get('prefarea', type=str),
            'furnishingstatus': form_data.get('furnishingstatus', type=str)
        }

        # 2. Basic validation
        required_fields = list(raw_input_values.keys())
        missing_fields = [field for field in required_fields if raw_input_values.get(field) is None]

        if missing_fields:
            # More specific error message
            error_msg = f"Error: Missing or invalid input value(s) for: {', '.join(missing_fields)}. Please check these fields."
            flash(error_msg, "error")
            return render_template('index.html', request=form_data) # Pass original form data back

        # 3. Create a copy for processing and map binary features
        processed_input_values = raw_input_values.copy()
        binary_features_in_form = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
        print("\nMapping binary features from form...")
        for feature in binary_features_in_form:
            if feature in processed_input_values:
                original_value = processed_input_values[feature]
                processed_input_values[feature] = 1 if original_value == 'yes' else 0
                # print(f"  Mapped '{feature}': '{original_value}' -> {processed_input_values[feature]}") # Debug print
            else:
                print(f"Warning: Binary feature '{feature}' not found in form data.")

        # 4. Convert the *processed* dictionary to a pandas DataFrame
        input_df = pd.DataFrame([processed_input_values])
        print("\nInput DataFrame for Prediction (after mapping):")
        print(input_df[binary_features_in_form + ['furnishingstatus']].head()) # Show mapped/categorical part

        # 5. Make prediction using the loaded pipeline
        print("Calling pipeline.predict()...")
        prediction = pipeline.predict(input_df)
        predicted_price = prediction[0]

        # 6. Format the result
        prediction_result_text = f"Predicted House Price: ₹ {predicted_price:,.2f}"
        print(f"Prediction successful: {prediction_result_text}")

    except ValueError as ve:
        error_msg = f"Error: Invalid input data type. Please ensure all numerical fields contain only numbers. Detail: {ve}"
        print(f"Prediction Error: {error_msg}")
        flash(error_msg, "error")
    except KeyError as ke:
         error_msg = f"Error: A required feature ({ke}) was missing during the prediction process. This might indicate an issue with the trained model or input data structure."
         print(f"Prediction Error: {error_msg}")
         flash(error_msg, "error")
    except Exception as e:
        error_msg = f"An unexpected error occurred during prediction: {e}"
        print(f"Prediction Error: {error_msg}")
        traceback.print_exc() # Log full traceback server-side
        flash("An unexpected error occurred. Please try again or check server logs.", "error") # User-friendly message

    # 7. Render the template again, passing prediction result and ORIGINAL form data
    return render_template('index.html',
                           prediction_text=prediction_result_text,
                           request=form_data)


# --- Run the App ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    # Use '0.0.0.0' to make accessible on network (needed for Render)
    host = os.environ.get('HOST', '0.0.0.0')
    # Default debug to False for safety, can be overridden by env variable
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() in ['true', '1', 't']

    print(f"--- Flask App Running ---")
    print(f"Model Status: {'Loaded' if pipeline else 'Not Loaded/Error'}")
    if pipeline_load_error and not pipeline: print(f"Load Error: {pipeline_load_error}")
    print(f"Access URL: http://{host}:{port} (or http://127.0.0.1:{port} from local machine)")
    print(f"Debug mode: {debug_mode}")
    print("Press CTRL+C to quit")
    app.run(host=host, port=port, debug=debug_mode)

requirements.txt

# requirements.txt (Updated & Minimal)

Flask==3.1.0
gunicorn==23.0.0
joblib==1.2.0
numpy==1.24.2
pandas==1.5.3
scikit-learn==1.2.2

# Notes:
# - numpy was pinned to 1.24.2 for compatibility with pandas 1.5.3 on Render.
# - scipy will be installed automatically as a dependency.
# - Werkzeug, Jinja2, etc., will be installed as dependencies of Flask.

# render.yaml
services:
  - type: web        # Type of service
    name: house-price-predictor-flask # <<< Make sure this matches your Render service name
    env: python      # Environment runtime
    repo: https://github.com/sahilsghadg/house-price-predictor-flask.git # <<< Your GitHub repo URL
    branch: main     # Your deployment branch
    plan: free       # Render plan
    build: # Build configuration
      # System packages needed to build some Python packages (like numpy/scipy if wheels aren't available)
      packages:
        - build-essential
        - python3-dev
        - libblas-dev
        - liblapack-dev
    buildCommand: "pip install --upgrade pip && pip install -r requirements.txt" # Update pip, then install project deps
    startCommand: "gunicorn app:app" # How to start the web server
    # Optional: Specify Python version
    # pythonVersion: "3.11" # Choose a version supported by Render and your code
    # Optional: Environment variables
    # envVars:
    #  - key: FLASK_DEBUG
    #    value: 0 # Set to 0 for production

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>House Price Predictor</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; margin: 0; padding: 0; background-color: #f8f9fa; color: #343a40; line-height: 1.6; }
        .navbar { background-color: #007bff; padding: 10px 20px; color: white; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .navbar h1 { margin: 0; font-size: 1.5em; }
        .container { max-width: 750px; margin: 30px auto; background: #ffffff; padding: 25px 40px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.08); }
        h2 { text-align: center; color: #495057; margin-bottom: 25px; font-weight: 500; }
        label { display: block; margin-top: 15px; margin-bottom: 5px; font-weight: bold; color: #495057; font-size: 0.95em;}
        input[type=number], select {
            width: 100%;
            padding: 12px;
            margin-top: 5px;
            border: 1px solid #ced4da;
            border-radius: 5px;
            box-sizing: border-box;
            background-color: #f1f3f5;
            transition: border-color 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
            font-size: 1em;
        }
        input[type=number]:focus, select:focus {
             border-color: #80bdff; outline: 0; box-shadow: 0 0 0 0.2rem rgba(0,123,255,.25); background-color: #fff;
        }
        input[type=submit] {
            background-color: #28a745; /* Success green */
            color: white;
            padding: 14px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1.1em;
            font-weight: bold;
            margin-top: 30px;
            width: 100%;
            transition: background-color 0.2s ease-in-out, transform 0.1s ease;
        }
        input[type=submit]:hover { background-color: #218838; transform: translateY(-1px); }
        input[type=submit]:active { transform: translateY(0px); }
        .prediction-box {
            margin-top: 30px; padding: 20px; background-color: #e9f7ef; border: 1px solid #a6d9b8; color: #155724; border-radius: 5px; text-align: center; font-size: 1.25em; font-weight: bold;
        }
        .flash-error, .flash-warning {
            padding: 12px 15px; margin-bottom: 15px; border-radius: 5px; border: 1px solid transparent;
        }
        .flash-error { background-color: #f8d7da; color: #721c24; border-color: #f5c6cb; }
        .flash-warning { background-color: #fff3cd; color: #856404; border-color: #ffeeba; }
        hr { border: 0; height: 1px; background: #e9ecef; margin: 30px 0; }
        .grid-container {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px 25px; /* row-gap column-gap */
        }
        footer { text-align: center; margin-top: 40px; padding: 15px; color: #6c757d; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="navbar">
        <h1>🏠 House Price Predictor</h1>
    </div>

    <div class="container">
        <h2>Enter House Details</h2>

        <!-- Flash messages -->
        {% with messages = get_flashed_messages(with_categories=true) %}
          {% if messages %}
            {% for category, message in messages %}
              {# Default category is 'message', treat it as 'warning' if not error #}
              {% set category = category if category == 'error' else 'warning' %}
              <div class="flash-{{ category }}">{{ message }}</div>
            {% endfor %}
          {% endif %}
        {% endwith %}

        <form action="/predict" method="post">
            <div class="grid-container">
                <div class="grid-item">
                    <label for="area">Area (in sqft):</label>
                    <!-- Access form data directly from request object passed to template -->
                    <input type="number" id="area" name="area" value="{{ request.area if request else 5000 }}" min="1000" max="20000" step="50" required>
                </div>
                <div class="grid-item">
                    <label for="bedrooms">Bedrooms:</label>
                    <input type="number" id="bedrooms" name="bedrooms" value="{{ request.bedrooms if request else 3 }}" min="1" max="6" step="1" required>
                </div>
                 <div class="grid-item">
                    <label for="bathrooms">Bathrooms:</label>
                    <input type="number" id="bathrooms" name="bathrooms" value="{{ request.bathrooms if request else 2 }}" min="1" max="4" step="1" required>
                </div>
                 <div class="grid-item">
                    <label for="stories">Stories (Floors):</label>
                    <input type="number" id="stories" name="stories" value="{{ request.stories if request else 2 }}" min="1" max="4" step="1" required>
                </div>
                 <div class="grid-item">
                    <label for="parking">Parking Spaces:</label>
                    <input type="number" id="parking" name="parking" value="{{ request.parking if request else 1 }}" min="0" max="4" step="1" required>
                </div>

                 <div class="grid-item">
                    <label for="mainroad">Main Road Access:</label>
                    <select id="mainroad" name="mainroad">
                        <option value="yes" {% if request and request.mainroad == 'yes' %}selected{% elif not request %}selected{% endif %}>Yes</option>
                        <option value="no" {% if request and request.mainroad == 'no' %}selected{% endif %}>No</option>
                    </select>
                </div>
                 <div class="grid-item">
                    <label for="guestroom">Guest Room Available:</label>
                    <select id="guestroom" name="guestroom">
                        <option value="yes" {% if request and request.guestroom == 'yes' %}selected{% endif %}>Yes</option>
                        <option value="no" {% if request and request.guestroom == 'no' %}selected{% elif not request %}selected{% endif %}>No</option>
                    </select>
                </div>
                 <div class="grid-item">
                    <label for="basement">Basement Available:</label>
                    <select id="basement" name="basement">
                         <option value="yes" {% if request and request.basement == 'yes' %}selected{% endif %}>Yes</option>
                        <option value="no" {% if request and request.basement == 'no' %}selected{% elif not request %}selected{% endif %}>No</option>
                    </select>
                </div>
                 <div class="grid-item">
                    <label for="hotwaterheating">Hot Water Heating:</label>
                    <select id="hotwaterheating" name="hotwaterheating">
                        <option value="yes" {% if request and request.hotwaterheating == 'yes' %}selected{% endif %}>Yes</option>
                        <option value="no" {% if request and request.hotwaterheating == 'no' %}selected{% elif not request %}selected{% endif %}>No</option>
                    </select>
                </div>
                 <div class="grid-item">
                    <label for="airconditioning">Air Conditioning:</label>
                    <select id="airconditioning" name="airconditioning">
                        <option value="yes" {% if request and request.airconditioning == 'yes' %}selected{% elif not request %}selected{% endif %}>Yes</option>
                        <option value="no" {% if request and request.airconditioning == 'no' %}selected{% endif %}>No</option>
                    </select>
                </div>
                 <div class="grid-item">
                    <label for="prefarea">Located in Preferred Area:</label>
                    <select id="prefarea" name="prefarea">
                         <option value="yes" {% if request and request.prefarea == 'yes' %}selected{% endif %}>Yes</option>
                        <option value="no" {% if request and request.prefarea == 'no' %}selected{% elif not request %}selected{% endif %}>No</option>
                    </select>
                </div>
                 <div class="grid-item">
                    <label for="furnishingstatus">Furnishing Status:</label>
                    <select id="furnishingstatus" name="furnishingstatus">
                        <option value="furnished" {% if request and request.furnishingstatus == 'furnished' %}selected{% endif %}>Furnished</option>
                        <option value="semi-furnished" {% if request and request.furnishingstatus == 'semi-furnished' %}selected{% elif not request %}selected{% endif %}>Semi-furnished</option>
                        <option value="unfurnished" {% if request and request.furnishingstatus == 'unfurnished' %}selected{% endif %}>Unfurnished</option>
                    </select>
                </div>
            </div> <!-- End Grid Container -->

            <input type="submit" value="Predict Price">
        </form>

        <!-- Display prediction result here -->
        {% if prediction_text %}
        <hr>
        <div class="prediction-box">
            {{ prediction_text }}
        </div>
        {% endif %}

    </div>

    <footer>
        <p>House Price Predictor - Flask Demo</p>
    </footer>
</body>
</html>