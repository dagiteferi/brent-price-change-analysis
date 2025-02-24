import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

class OilPricePredictor:
    def __init__(self):
        """Initialize the OilPricePredictor class."""
        self.merged_data = None  # Merged dataset
        self.feature_data = None  # Dataset with engineered features
        self.X = None  # Features for model training
        self.y = None  # Target variable (oil price)
        self.setup_logging()  # Configure logging

    def setup_logging(self):
        """Configure logging settings to save logs in 'logs/Adapting.log'."""
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)  # Create logs directory if it doesn't exist

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_dir / "Adapting.log"),  # Save logs to file
                logging.StreamHandler()  # Print logs to console
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Logging initialized.")

    def load_data(self, data_path="../data"):
        """Load and prepare the data from CSV files."""
        try:
            self.logger.info("Loading data from CSV files...")

            # Load individual datasets
            gdp_data_daily = pd.read_csv(f"{data_path}/GDP_cleaned_data_daily.csv")
            gdp_data_daily['Date'] = pd.to_datetime(gdp_data_daily['Date'])
            gdp_data_daily.set_index('Date', inplace=True)
            
            cpi_data_daily = pd.read_csv(f"{data_path}/CPI_cleaned_data_daily.csv")
            cpi_data_daily['Date'] = pd.to_datetime(cpi_data_daily['Date'])
            cpi_data_daily.set_index('Date', inplace=True)
            
            exchange_rate_data_daily = pd.read_csv(f"{data_path}/Exchange_Rate_cleaned_data_daily.csv")
            exchange_rate_data_daily['Date'] = pd.to_datetime(exchange_rate_data_daily['Date'])
            exchange_rate_data_daily.set_index('Date', inplace=True)
            
            oil_data_daily = pd.read_csv(f"{data_path}/Copy of BrentOilPrices.csv")
            oil_data_daily['Date'] = pd.to_datetime(oil_data_daily['Date'])
            oil_data_daily.set_index('Date', inplace=True)
            
            self.logger.info("Data loaded successfully!")

            # Merge data
            self.merged_data = self.merge_data(oil_data_daily, gdp_data_daily, 
                                             cpi_data_daily, exchange_rate_data_daily)
            self.feature_data = self.create_features(self.merged_data)
            
            # Prepare features and target
            self.X = self.feature_data[['GDP', 'CPI', 'Exchange_Rate', 'Price_Pct_Change',
                                      'GDP_Pct_Change', 'CPI_Pct_Change', 'Exchange_Rate_Pct_Change',
                                      'Price_MA7', 'Price_MA30', 'Price_Volatility']]
            self.y = self.feature_data['Price']
            
            self.logger.info("Data preprocessing completed.")
            return True
            
        except FileNotFoundError as e:
            self.logger.error(f"Error loading data: {e}")
            self.logger.error("Please ensure all required CSV files are in the correct directory.")
            return False

    def merge_data(self, oil_data, gdp_data, cpi_data, exchange_rate_data):
        """Merge all datasets into a single DataFrame."""
        self.logger.info("Merging datasets...")
        merged_data = pd.concat([oil_data, gdp_data, cpi_data, exchange_rate_data], 
                              axis=1, join='inner')
        merged_data.columns = ['Price', 'GDP', 'CPI', 'Exchange_Rate']
        return merged_data.dropna()

    def create_features(self, data):
        """Create additional features from the merged data."""
        self.logger.info("Creating features...")
        feature_data = data.copy()
        
        # Calculate percentage changes
        feature_data['Price_Pct_Change'] = feature_data['Price'].pct_change()
        feature_data['GDP_Pct_Change'] = feature_data['GDP'].pct_change()
        feature_data['CPI_Pct_Change'] = feature_data['CPI'].pct_change()
        feature_data['Exchange_Rate_Pct_Change'] = feature_data['Exchange_Rate'].pct_change()
        
        # Moving averages
        feature_data['Price_MA7'] = feature_data['Price'].rolling(window=7).mean()
        feature_data['Price_MA30'] = feature_data['Price'].rolling(window=30).mean()
        
        # Volatility
        feature_data['Price_Volatility'] = feature_data['Price'].rolling(window=30).std()
        
        return feature_data.dropna()

    def build_lstm_model(self, X_train, y_train):
        """Build and train the LSTM model."""
        try:
            self.logger.info("Building LSTM model...")

            # Scale the data
            X_scaler = MinMaxScaler()
            y_scaler = MinMaxScaler()
            
            X_scaled = X_scaler.fit_transform(X_train)
            y_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
            
            # Reshape input for LSTM
            X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
            
            # Build the LSTM model
            model = Sequential([
                LSTM(50, activation='relu', input_shape=(1, X_train.shape[1])),
                Dense(1)
            ])
            
            # Compile the model
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            # Train the model
            self.logger.info("Training LSTM model...")
            model.fit(X_lstm, y_scaled, epochs=100, batch_size=32, verbose=0)
            
            self.logger.info("LSTM model training completed.")
            return model, X_scaler, y_scaler
        except Exception as e:
            self.logger.error(f"Error in LSTM model building: {str(e)}")
            return None, None, None

    def train_and_evaluate(self, n_splits=5):
        """Train and evaluate the LSTM model using time series cross-validation."""
        if self.X is None or self.y is None:
            self.logger.error("Data not loaded. Please load data first using load_data().")
            return None

        self.logger.info("Starting time series cross-validation...")
        tscv = TimeSeriesSplit(n_splits=n_splits)
        lstm_scores = []

        for train_index, test_index in tscv.split(self.X):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]

            # LSTM Model
            lstm_model, X_scaler, y_scaler = self.build_lstm_model(X_train, y_train)
            if lstm_model is not None:
                # Prepare test data
                X_test_scaled = X_scaler.transform(X_test)
                X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
                
                # Make predictions
                lstm_forecast = lstm_model.predict(X_test_lstm)
                lstm_forecast = y_scaler.inverse_transform(lstm_forecast)
                
                # Evaluate performance
                mse = mean_squared_error(y_test, lstm_forecast.flatten())
                lstm_scores.append(mse)
                self.logger.info(f"Fold MSE: {mse:.4f}")

        # Calculate average MSE
        avg_lstm_mse = np.mean(lstm_scores) if lstm_scores else None

        # Print evaluation results
        self.logger.info("\nCross-Validation Model Evaluation Results:")
        if avg_lstm_mse is not None:
            self.logger.info(f"LSTM - Average MSE: {avg_lstm_mse:.4f}")

# Example usage
if __name__ == "__main__":
    predictor = OilPricePredictor()
    if predictor.load_data():
        predictor.train_and_evaluate(n_splits=5)