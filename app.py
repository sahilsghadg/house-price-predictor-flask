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

        # 3. Create a copy for processing and map binary features (FIX 1)
        processed_input_values = raw_input_values.copy()
        binary_features_in_form = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
        print("\nMapping binary features from form...")
        for feature in binary_features_in_form:
            if feature in processed_input_values:
                original_value = processed_input_values[feature]
                processed_input_values[feature] = 1 if original_value == 'yes' else 0
                # print(f"  Mapped '{feature}': '{original_value}' -> {processed_input_values[feature]}") # Debug print
            else:
                # This case shouldn't happen if form matches keys, but good for robustness
                print(f"Warning: Binary feature '{feature}' not found in form data.")

        # 4. Convert the *processed* dictionary to a pandas DataFrame
        # Ensure columns match expected order if your pipeline is sensitive (less likely with ColumnTransformer by name)
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
        # Don't reset prediction_result_text, let it be None
    except KeyError as ke:
         error_msg = f"Error: A required feature ({ke}) was missing during the prediction process. This might indicate an issue with the trained model or input data structure."
         print(f"Prediction Error: {error_msg}")
         flash(error_msg, "error")
    except Exception as e:
        error_msg = f"An unexpected error occurred during prediction: {e}"
        print(f"Prediction Error: {error_msg}")
        # Log the full traceback for debugging server-side
        traceback.print_exc()
        flash("An unexpected error occurred. Please try again or contact support.", "error") # User-friendly message

    # 7. Render the template again, passing prediction result and ORIGINAL form data (FIX 2)
    # Pass `form_data` (which is request.form) back to the template
    return render_template('index.html',
                           prediction_text=prediction_result_text,
                           request=form_data)


# --- Run the App ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    # Use '0.0.0.0' to make it accessible on your local network (needed for Render deployment)
    host = os.environ.get('HOST', '0.0.0.0')
    # Default debug to False for safety, can be overridden by environment variable
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() in ['true', '1', 't']

    print(f"--- Flask App Running ---")
    print(f"Model Status: {'Loaded' if pipeline else 'Not Loaded/Error'}")
    if pipeline_load_error and not pipeline: print(f"Load Error: {pipeline_load_error}")
    print(f"Access URL: http://{host}:{port} (or http://127.0.0.1:{port} from local machine)")
    print(f"Debug mode: {debug_mode}")
    print("Press CTRL+C to quit")
    # Use threaded=False if you experience issues with models in multithreaded environments (less common now)
    app.run(host=host, port=port, debug=debug_mode)