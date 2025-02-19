import os
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# Define the log directory and file path
log_dir = "../logs"
log_file = os.path.join(log_dir, "eda.log")

# Create the logs directory if it doesn't exist
os.makedirs(log_dir, exist_ok=True)

# Set up logging
logger = logging.getLogger('EDA Logger')
logger.setLevel(logging.DEBUG)

# Configure logging to write to both file and console
logging.basicConfig(
    level=logging.INFO,  # Log level
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),  # Save logs to file
        logging.StreamHandler()  # Print logs in console
    ]
)

def check_missing_values(data):
    """
    Check for missing values in the dataset.
    
    Parameters:
    -----------
    data (pd.DataFrame): The dataset to check.
    
    Returns:
    --------
    pd.Series: Number of missing values per column.
    """
    missing_values = data.isnull().sum()
    logger.info("Missing values per column:")
    logger.info(missing_values)
    return missing_values

def check_duplicates(data):
    """
    Check for duplicate rows in the dataset.
    
    Parameters:
    -----------
    data (pd.DataFrame): The dataset to check.
    
    Returns:
    --------
    int: Number of duplicate rows.
    """
    duplicates = data.duplicated().sum()
    logger.info(f"Number of duplicate rows: {duplicates}")
    return duplicates

def check_outliers(data, column="Price"):
    """
    Check for outliers in a specific column using the IQR method.
    
    Parameters:
    -----------
    data (pd.DataFrame): The dataset to check.
    column (str): The column to check for outliers.
    
    Returns:
    --------
    pd.DataFrame: Rows identified as outliers.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    logger.info(f"Number of outliers in '{column}': {len(outliers)}")
    return outliers

def save_cleaned_data(data, file_path):
    """
    Save the cleaned dataset to a file.
    
    Parameters:
    -----------
    data (pd.DataFrame): The cleaned dataset.
    file_path (str): Path to save the cleaned dataset.
    """
    try:
        logger.info(f"Saving cleaned data to: {file_path}")
        data.to_csv(file_path)
        logger.info("Cleaned data saved successfully.")
    except Exception as e:
        logger.error(f"Error saving cleaned data: {e}")
        raise e





def plot_historical_prices(data):
    """
    Plot historical Brent oil prices.
    
    Parameters:
    -----------
    data (pd.DataFrame): The dataset containing the 'Price' column.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data["Price"], label="Brent Oil Prices", color="blue")
    plt.title("Historical Brent Oil Prices (1987 - 2022)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD per barrel)")
    plt.legend()
    plt.grid()
    plt.show()

def plot_rolling_statistics(data):
    """
    Plot rolling mean and standard deviation of Brent oil prices.
    
    Parameters:
    -----------
    data (pd.DataFrame): The dataset containing the 'Price' column.
    """
    # Calculate rolling mean and standard deviation
    data["Rolling Mean"] = data["Price"].rolling(window=30).mean()
    data["Rolling Std"] = data["Price"].rolling(window=30).std()

    # Plot rolling statistics
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data["Price"], label="Brent Oil Prices", color="blue")
    plt.plot(data.index, data["Rolling Mean"], label="Rolling Mean", color="red")
    plt.plot(data.index, data["Rolling Std"], label="Rolling Std", color="green")
    plt.title("Rolling Mean and Standard Deviation of Brent Oil Prices")
    plt.xlabel("Date")
    plt.ylabel("Price (USD per barrel)")
    plt.legend()
    plt.grid()
    plt.show()

def check_stationarity(data, column="Price"):
    """
    Check for stationarity using the Augmented Dickey-Fuller (ADF) test.
    
    Parameters:
    -----------
    data (pd.DataFrame): The dataset containing the column to check.
    column (str): The column to check for stationarity.
    
    Returns:
    --------
    dict: ADF test results.
    """
    result = adfuller(data[column])
    logger.info(f"ADF Statistic: {result[0]}")
    logger.info(f"p-value: {result[1]}")
    logger.info(f"Critical Values: {result[4]}")
    return {
        "ADF Statistic": result[0],
        "p-value": result[1],
        "Critical Values": result[4]
    }




def plot_price_trend_over_years(data):
    """
    Plot a bar chart showing the average Brent oil price trend over the years.
    
    Parameters:
    -----------
    data (pd.DataFrame): The dataset containing the 'Price' column with 'Date' as the index.
    """
    # Ensure the index is datetime
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    
    # Aggregate average price per year
    yearly_avg_price = data["Price"].resample("Y").mean().reset_index()
    yearly_avg_price["Year"] = yearly_avg_price["Date"].dt.year

    # Plot the bar chart using Seaborn
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Year', y='Price', data=yearly_avg_price, palette='viridis')
    plt.title('Average Yearly Brent Oil Prices', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Average Price (USD per barrel)', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()




def plot_with_events(data):
    """
    Plot Brent oil prices with significant event markers as vertical dashed lines.
    
    Parameters:
    -----------
    data (pd.DataFrame): The dataset containing the 'Price' column with 'Date' as the index.
    """

    # Dictionary of significant events
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

    plt.figure(figsize=(14, 7))

    # Plot Brent oil prices using the index (Date)
    plt.plot(data.index, data['Price'], label='Brent Oil Price', color='blue')

    # Add vertical dashed lines for events
    for date, event in significant_events.items():
        event_date = pd.to_datetime(date)
        plt.axvline(event_date, color='r', linestyle='--', linewidth=1.5, label=f'{event} ({date})')

    # Formatting
    plt.title('Brent Oil Price Over Time with Event Markers')
    plt.xlabel('Date')
    plt.ylabel('Price (USD per barrel)')
    plt.legend(loc='best', fontsize=9)
    plt.grid()

    plt.show()