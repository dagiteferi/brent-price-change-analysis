import pandas as pd
import matplotlib.pyplot as plt
import logging
import os
import numpy as np
import seaborn as sns
from datetime import timedelta
from scipy import stats
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
# Set up logging
log_file_path = 'logs/analysis.log'
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)



# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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

    Parameters:
    -----------
    df (pd.DataFrame): The dataset containing the 'Price' column with 'Date' as the index.
    event_date (pd.Timestamp): The date of the event.
    days_before (int): Number of days before the event to include.
    days_after (int): Number of days after the event to include.

    Returns:
    --------
    pd.DataFrame: A subset of the DataFrame containing prices around the event.
    """
    start_date = event_date - timedelta(days=days_before)
    end_date = event_date + timedelta(days=days_after)
    return df.loc[start_date:end_date]

def analyze_events(df):
    """
    Analyze the impact of significant events on Brent oil prices.

    Parameters:
    -----------
    df (pd.DataFrame): The dataset containing the 'Price' column with 'Date' as the index.
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

        # Calculate cumulative returns
        if not prices_around_event.empty:
            try:
                prices_before = prices_around_event.loc[:event_date]
                prices_after = prices_around_event.loc[event_date:]

                cum_return_before = prices_before['Price'].pct_change().add(1).cumprod().iloc[-1] - 1
                cum_return_after = prices_after['Price'].pct_change().add(1).cumprod().iloc[-1] - 1
            except Exception as e:
                logging.error(f"Error calculating cumulative returns for {event_name}: {e}")
                cum_return_before = None
                cum_return_after = None
        else:
            cum_return_before = None
            cum_return_after = None

        # Store results
        results.append({
            "Event": event_name,
            "Date": date_str,
            "Change_1M": change_1m,
            "Change_3M": change_3m,
            "Change_6M": change_6m,
            "Cumulative Return Before": cum_return_before,
            "Cumulative Return After": cum_return_after
        })

    # Create DataFrame and log results
    event_impact_df = pd.DataFrame(results)
    logging.info("Event Impact Analysis: \n%s", event_impact_df)

    # Visualize results
    visualize_event_impact(event_impact_df, df)

def visualize_event_impact(event_impact_df, df):
    """
    Visualize the impact of significant events on Brent oil prices.

    Parameters:
    -----------
    event_impact_df (pd.DataFrame): The results of the event impact analysis.
    df (pd.DataFrame): The dataset containing the 'Price' column with 'Date' as the index.
    """
    # Line plot for price trends around events
    plt.figure(figsize=(14, 8))
    for date_str, event_name in significant_events.items():
        event_date = pd.to_datetime(date_str)
        prices_around_event = get_prices_around_event(df, event_date)

        if not prices_around_event.empty:
            plt.plot(prices_around_event.index, prices_around_event['Price'], label=f"{event_name} ({date_str})")
            plt.axvline(event_date, color='red', linestyle='--', linewidth=0.8)
            plt.text(event_date, prices_around_event['Price'].max(), event_name, 
                     rotation=90, verticalalignment='bottom', fontsize=8)

    plt.title("Brent Oil Price Trends Around Key Events")
    plt.xlabel("Date")
    plt.ylabel("Price (USD per barrel)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # Bar plot for percentage changes
    changes_data = event_impact_df.melt(id_vars=["Event", "Date"], 
                                          value_vars=["Change_1M", "Change_3M", "Change_6M"])
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    sns.barplot(data=changes_data, x="Event", y="value", hue="variable", ax=axes[0])
    axes[0].set_title("Percentage Change in Brent Oil Prices Before and After Events")
    axes[0].set_ylabel("Percentage Change")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
    axes[0].legend(title="Change Period")

    # Bar plot for cumulative returns
    returns_data = event_impact_df.melt(id_vars=["Event", "Date"], 
                                          value_vars=["Cumulative Return Before", "Cumulative Return After"])
    sns.barplot(data=returns_data, x="Event", y="value", hue="variable", ax=axes[1])
    axes[1].set_title("Cumulative Returns Before and After Events")
    axes[1].set_ylabel("Cumulative Return")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
    axes[1].legend(title="Return Period")

    plt.tight_layout()
    plt.show()

    # Perform statistical analysis
    statistical_analysis(df)

def statistical_analysis(df):
    """
    Perform statistical analysis (t-test) to compare prices before and after events.

    Parameters:
    -----------
    df (pd.DataFrame): The dataset containing the 'Price' column with 'Date' as the index.
    """
    t_test_results = {}
    for date_str, event_name in significant_events.items():
        event_date = pd.to_datetime(date_str)
        prices_around = get_prices_around_event(df, event_date)

        if not prices_around.empty:
            before_prices = prices_around.loc[:event_date]['Price']
            after_prices = prices_around.loc[event_date:]['Price']

            if len(before_prices) > 1 and len(after_prices) > 1:
                t_stat, p_val = stats.ttest_ind(before_prices, after_prices, nan_policy='omit')
                t_test_results[event_name] = {"t-statistic": t_stat, "p-value": p_val}
            else:
                t_test_results[event_name] = {"t-statistic": None, "p-value": None}

    t_test_df = pd.DataFrame(t_test_results).T
    logging.info("T-Test Results: \n%s", t_test_df)
    print("\nT-Test Results:")
    print(t_test_df)


def process_data(data):
    """
    Difference the series to make it stationary.
    
    Parameters:
    data (DataFrame): The input data containing the 'Price' column.
    
    Returns:
    Series: Differenced data.
    """
    logging.info('Differencing the data to make it stationary.')
    diff_data = data['Price'].diff().dropna()
    logging.info('Data differenced successfully.')
    return diff_data
def fit_markov_switching_model(diff_data):
    """
    Fit a Markov-Switching AR model (2 regimes, AR(1)).
    
    Parameters:
    diff_data (Series): The differenced data.
    
    Returns:
    results (Result): The fitted model results.
    """
    try:
        logging.info('Fitting the Markov-Switching AR model.')
        model = MarkovAutoregression(diff_data, k_regimes=2, order=1, switching_ar=True)
        results = model.fit()
        logging.info('Model fitted successfully.')
        return results
    except Exception as e:
        logging.error(f'Error fitting the model: {e}')
        return None
def preprocess_data(data):
    """
    Preprocess the data for LSTM:
    1. Scale the 'Price' column to a range of [0, 1].
    2. Split the data into training and testing sets.
    3. Create sequences for LSTM input.
    """
    logger.info("Preprocessing data for LSTM...")
    
    # Scale the 'Price' column
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Price']].values)
    
    # Split the data into training and testing sets
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]
    
    # Create sequences for LSTM input
    def create_dataset(data, time_step=60):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)
    
    time_step = 60
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    
    # Reshape input to be [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    return X_train, y_train, X_test, y_test, scaler, train_size, time_step

def build_lstm_model(input_shape):
    """
    Build and compile the LSTM model.
    """
    logger.info("Building LSTM model...")
    
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

def train_model(model, X_train, y_train, epochs=10, batch_size=32):
    """
    Train the LSTM model.
    """
    logger.info("Training LSTM model...")
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
    return history

def evaluate_model(model, X_test, y_test, scaler):
    """
    Evaluate the LSTM model on the test data and calculate metrics.
    """
    logger.info("Evaluating LSTM model...")
    
    # Make predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)  # Scale back to original range
    
    # Scale y_test back to original range
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate evaluation metrics
    mse = mean_squared_error(y_test_original, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_original, predictions)
    
    logger.info(f"Mean Squared Error (MSE): {mse}")
    logger.info(f"Root Mean Squared Error (RMSE): {rmse}")
    logger.info(f"Mean Absolute Error (MAE): {mae}")
    
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    
    return predictions, mse, rmse, mae

