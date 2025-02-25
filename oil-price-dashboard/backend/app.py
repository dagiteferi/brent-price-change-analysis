from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import numpy as np
from pathlib import Path
import logging
from datetime import timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Set up logging
log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)  # Create logs directory if it doesn't exist
log_file = log_dir / "app.log"  # Log file name based on the script name

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),  # Save logs to file
        logging.StreamHandler()  # Print logs to console
    ]
)
logger = logging.getLogger(__name__)
logger.info("Logging initialized.")

# Load the LSTM model and scalers
try:
    logger.info("Loading LSTM model and scalers...")
    model = load_model("models/lstm_model.h5")
    X_scaler = joblib.load("models/X_scaler.pkl")
    y_scaler = joblib.load("models/y_scaler.pkl")
    logger.info("Model and scalers loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model or scalers: {str(e)}")
    raise

# Load datasets
data_dir = Path("data")
try:
    logger.info("Loading datasets...")
    oil_data = pd.read_csv(data_dir / "BrentOilPrices.csv")
    oil_data['Date'] = pd.to_datetime(oil_data['Date'])
    oil_data.set_index('Date', inplace=True)
    logger.info("Datasets loaded successfully.")
except Exception as e:
    logger.error(f"Error loading datasets: {str(e)}")
    raise

# Define significant events
significant_events = {
    '1990-08-02': 'Start-Gulf War',
    '1991-02-28': 'End-Gulf War',
    '2001-09-11': '9/11 Terrorist Attacks',
    '2003-03-20': 'Invasion of Iraq',
    '2005-07-07': 'London Terrorist Attack',
    '2010-12-18': 'Start-Arab Spring',
    '2011-02-17': 'Civil War in Libya Start',
    '2015-11-13': 'Paris Terrorist Attacks',
    '2019-12-31': 'Attack on US Embassy in Iraq',
    '2022-02-24': 'Russian Invasion of Ukraine',
}

def get_prices_around_event(df, event_date, days_before=180, days_after=180):
    """
    Extract Brent oil prices for a specified period before and after an event.
    """
    start_date = event_date - timedelta(days=days_before)
    end_date = event_date + timedelta(days=days_after)
    return df.loc[start_date:end_date]

def analyze_events(df):
    """
    Analyze the impact of significant events on Brent oil prices.
    """
    results = []
    for date_str, event_name in significant_events.items():
        event_date = pd.to_datetime(date_str)
        prices_around_event = get_prices_around_event(df, event_date)

        # Calculate percentage changes
        try:
            nearest_before_1m = df.index[df.index <= event_date - timedelta(days=30)][-1]
            nearest_after_1m = df.index[df.index >= event_date + timedelta(days=30)][0]
            price_before_1m = df.loc[nearest_before_1m, 'Price']
            price_after_1m = df.loc[nearest_after_1m, 'Price']
            change_1m = ((price_after_1m - price_before_1m) / price_before_1m) * 100
        except (IndexError, KeyError):
            change_1m = None

        try:
            nearest_before_3m = df.index[df.index <= event_date - timedelta(days=90)][-1]
            nearest_after_3m = df.index[df.index >= event_date + timedelta(days=90)][0]
            price_before_3m = df.loc[nearest_before_3m, 'Price']
            price_after_3m = df.loc[nearest_after_3m, 'Price']
            change_3m = ((price_after_3m - price_before_3m) / price_before_3m) * 100
        except (IndexError, KeyError):
            change_3m = None

        try:
            nearest_before_6m = df.index[df.index <= event_date - timedelta(days=180)][-1]
            nearest_after_6m = df.index[df.index >= event_date + timedelta(days=180)][0]
            price_before_6m = df.loc[nearest_before_6m, 'Price']
            price_after_6m = df.loc[nearest_after_6m, 'Price']
            change_6m = ((price_after_6m - price_before_6m) / price_before_6m) * 100
        except (IndexError, KeyError):
            change_6m = None

        # Store results
        results.append({
            "Event": event_name,
            "Date": date_str,
            "Change_1M": change_1m,
            "Change_3M": change_3m,
            "Change_6M": change_6m,
        })

    return results

@app.route('/api/data', methods=['GET'])
def get_data():
    """Return historical oil price data, optionally filtered by date range."""
    try:
        logger.info("Received request for historical data.")
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        filtered_data = oil_data.copy()

        if start_date and end_date:
            logger.info(f"Filtering data from {start_date} to {end_date}.")
            filtered_data = filtered_data[
                (filtered_data.index >= pd.to_datetime(start_date)) & (filtered_data.index <= pd.to_datetime(end_date))
            ]

        logger.info("Returning filtered data.")
        return jsonify(filtered_data.reset_index().to_dict(orient="records"))
    except Exception as e:
        logger.error(f"Error in /api/data: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/events', methods=['GET'])
def get_events():
    """Return event analysis results."""
    try:
        logger.info("Received request for event analysis.")
        event_results = analyze_events(oil_data)
        logger.info("Returning event analysis results.")
        return jsonify(event_results)
    except Exception as e:
        logger.error(f"Error in /api/events: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make predictions using the LSTM model."""
    try:
        logger.info("Received prediction request.")
        data = request.get_json()
        input_data = np.array([[
            data['GDP'],
            data['CPI'],
            data['Exchange_Rate'],
            data['Price_Pct_Change'],
            data['GDP_Pct_Change'],
            data['CPI_Pct_Change'],
            data['Exchange_Rate_Pct_Change'],
            data['Price_MA7'],
            data['Price_MA30'],
            data['Price_Volatility']
        ]])

        # Preprocess the input data
        logger.info("Preprocessing input data.")
        input_data_scaled = X_scaler.transform(input_data)
        input_data_reshaped = input_data_scaled.reshape((input_data_scaled.shape[0], 1, input_data_scaled.shape[1]))

        # Make predictions
        logger.info("Making predictions.")
        predictions_scaled = model.predict(input_data_reshaped)
        predictions = y_scaler.inverse_transform(predictions_scaled)

        logger.info(f"Prediction successful: {predictions[0][0]}")
        return jsonify({"predicted_oil_price": predictions[0][0]})
    except Exception as e:
        logger.error(f"Error in /api/predict: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Return model performance metrics."""
    try:
        logger.info("Calculating model performance metrics.")
        
        # Example: Replace y_true and y_pred with actual values
        # These should be calculated during model evaluation
        y_true = [68.91, 69.15, 68.42]  # Actual oil prices
        y_pred = [68.95, 69.10, 68.50]  # Predicted oil prices

        # Calculate metrics
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        metrics = {
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        }
        logger.info("Returning model metrics.")
        return jsonify(metrics)
    except Exception as e:
        logger.error(f"Error in /api/metrics: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask app.")
    app.run(debug=True)