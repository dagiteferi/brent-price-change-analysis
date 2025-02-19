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






