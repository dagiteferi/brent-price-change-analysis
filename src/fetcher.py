import logging
import pandas as pd
import numpy as np
import wbdata
from pathlib import Path

class WorldBankDataFetcher:
    """
    A class to fetch and process economic indicator data from the World Bank.
    """
    
    def __init__(self, start_date, end_date):
        """
        Initialize the fetcher with a date range.
        
        Args:
            start_date (str): Start date for data retrieval (YYYY-MM-DD format).
            end_date (str): End date for data retrieval (YYYY-MM-DD format).
        """
        self.start_date = start_date
        self.end_date = end_date
        self.setup_logging()
        
    def setup_logging(self):
        """
        Configure logging settings to save logs in 'logs/fetcher.log'.
        """
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'fetcher.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Logging initialized.")
    
    def fetch_indicator_data(self, indicator_code, indicator_name, country='WLD'):
        """
        Fetch data for a specific economic indicator from the World Bank API.
        
        Args:
            indicator_code (str): World Bank indicator code.
            indicator_name (str): Descriptive name for the indicator.
            country (str): Country code (default: 'WLD' for global data).
        
        Returns:
            pd.DataFrame: Fetched and processed data.
        """
        try:
            self.logger.info(f"Fetching data for {indicator_name} ({indicator_code})...")
            data = wbdata.get_dataframe(
                {indicator_code: indicator_name},
                country=country,
                date=(self.start_date, self.end_date)
            )
            
            if data is not None and not data.empty:
                self.logger.info(f"Successfully fetched {indicator_name} data.")
            else:
                self.logger.warning(f"No data found for {indicator_name} ({indicator_code}).")
            
            return data
        except Exception as e:
            self.logger.error(f"Error fetching {indicator_name} data: {str(e)}")
            return None
    
    def process_data(self, df, indicator_name):
        """
        Clean and process the fetched data.
        
        Args:
            df (pd.DataFrame): Raw data DataFrame.
            indicator_name (str): Indicator name.
        
        Returns:
            pd.DataFrame: Cleaned and interpolated daily frequency data.
        """
        try:
            if df is None or df.empty:
                self.logger.warning(f"Skipping processing for {indicator_name}: No data available.")
                return pd.DataFrame()
            
            self.logger.info(f"Processing {indicator_name} data...")
            
            # Reset index and rename columns for clarity
            df = df.reset_index()
            df.columns = ['date', indicator_name]
            df['date'] = pd.to_datetime(df['date'])
            
            # Handle missing and infinite values
            df[indicator_name] = df[indicator_name].replace([np.inf, -np.inf], np.nan)
            df.dropna(inplace=True)
            
            # Resample data to daily frequency and interpolate missing values
            full_index = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')
            df_daily = df.set_index('date').reindex(full_index)
            df_daily.interpolate(method='cubic', inplace=True)
            df_daily.reset_index(inplace=True)
            df_daily.rename(columns={'index': 'Date'}, inplace=True)
            
            self.logger.info(f"Successfully processed {indicator_name} data.")
            return df_daily
        except Exception as e:
            self.logger.error(f"Error processing {indicator_name} data: {str(e)}")
            return pd.DataFrame()
    
    def save_data(self, df, indicator_name):
        """
        Save processed data to a CSV file.
        
        Args:
            df (pd.DataFrame): Processed data.
            indicator_name (str): Indicator name.
        """
        try:
            output_dir = Path("data")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{indicator_name}_cleaned_data_daily.csv"
            df.to_csv(output_path, index=False)
            self.logger.info(f"Successfully saved {indicator_name} data to {output_path}.")
        except Exception as e:
            self.logger.error(f"Error saving {indicator_name} data: {str(e)}")
    
def main():
    """
    Main function to fetch, process, and save economic indicator data.
    """
    indicators = {
        'NY.GDP.MKTP.CD': {'name': 'GDP', 'country': 'WLD'},
        'FP.CPI.TOTL.ZG': {'name': 'CPI', 'country': 'WLD'},
        'SL.UEM.TOTL.ZS': {'name': 'Unemployment_Rate', 'country': 'WLD'},
        'PA.NUS.FCRF': {'name': 'Exchange_Rate', 'country': 'EMU'}
    }
    
    fetcher = WorldBankDataFetcher('1987-05-20', '2022-11-14')
    
    for indicator_code, info in indicators.items():
        raw_data = fetcher.fetch_indicator_data(
            indicator_code,
            info['name'],
            info['country']
        )
        
        processed_data = fetcher.process_data(raw_data, info['name'])
        
        if not processed_data.empty:
            fetcher.save_data(processed_data, info['name'])

if __name__ == "__main__":
    main()
