import os
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller


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
    yearly_avg_price = data["Price"].resample("YE").mean().reset_index()
    yearly_avg_price["Year"] = yearly_avg_price["Date"].dt.year

    # Plot the bar chart using Seaborn
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Year', y='Price', data=yearly_avg_price, hue='Year', palette='viridis', dodge=False)
    plt.title('Average Yearly Brent Oil Prices', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Average Price (USD per barrel)', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.legend().set_visible(False)  # Hide the legend since 'hue' is used for colors only
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


from statsmodels.tsa.seasonal import seasonal_decompose

def plot_time_series_decomposition(data, column="Price", model="multiplicative"):
    """
    Decompose the time series into trend, seasonality, and residuals.
    
    Parameters:
    -----------
    data (pd.DataFrame): The dataset containing the column to analyze.
    column (str): The column to decompose.
    model (str): Type of decomposition ("additive" or "multiplicative").
    """
    decomposition = seasonal_decompose(data[column], model=model, period=365)  # Assuming daily data
    
    plt.figure(figsize=(12, 8))
    plt.subplot(4, 1, 1)
    plt.plot(data[column], label="Original")
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(decomposition.trend, label="Trend", color="red")
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(decomposition.seasonal, label="Seasonality", color="green")
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(decomposition.resid, label="Residuals", color="purple")
    plt.legend()

    plt.tight_layout()
    plt.show()