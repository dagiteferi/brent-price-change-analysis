import pandas as pd
import os
import logging
import sys

# Add the parent directory to the system path
sys.path.append(os.path.join(os.path.abspath('../')))

# Ensure the logs directory exists
def create_directory_if_not_exists(directory_path):
    """Helper function to create a directory if it doesn't already exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logging.info(f"Created directory: {directory_path}")
    else:
        logging.info(f"Directory already exists: {directory_path}")

# Set up logging
log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../logs"))
create_directory_if_not_exists(log_dir)

# Set up logging to a file
logging.basicConfig(
    filename=os.path.join(log_dir, 'data_loading.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Set up logging to the console as well
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add the console handler to the logger
logging.getLogger().addHandler(console_handler)

# Set the dataset directory
_DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
create_directory_if_not_exists(_DATASET_DIR)

def load_data(file_path, dataset_dir=_DATASET_DIR):
    """
    This function loads the data from the given file path and returns the data in a suitable format for the model.
    
    Parameters:
    -----------
    file_path (str): Path to the dataset file (relative to the _DATASET_DIR).
    dataset_dir (str): The root directory where the data is located. Default is _DATASET_DIR.
    
    Returns:
    --------
    pd.DataFrame: The loaded dataset as a pandas DataFrame.
    """
    full_file_path = os.path.join(dataset_dir, file_path)
    
    # Check if the file exists
    if not os.path.exists(full_file_path):
        logging.error(f"File not found: {full_file_path}")
        raise FileNotFoundError(f"File not found: {full_file_path}")
    
    try:
        logging.info(f"Attempting to load data from: {full_file_path}")
        data = pd.read_csv(full_file_path)
        logging.info(f"Data successfully loaded from: {full_file_path}")
        return data
    except Exception as e:
        logging.error(f"Error loading data from {full_file_path}: {e}")
        raise e