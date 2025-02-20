import pandas as pd
import matplotlib.pyplot as plt
import logging
import os
from datetime import timedelta

# Set up logging
log_file_path = 'logs/analysis.log'
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class OilPriceAnalyzer:
    def __init__(self, data):
        self.df = data.set_index(pd.to_datetime(data['Date'], format='%d-%b-%y'))
        self.significant_events = {
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
    
    def get_prices_around_event(self, event_date, days_before=180, days_after=180):
        start_date = event_date - timedelta(days=days_before)
        end_date = event_date + timedelta(days=days_after)
        return self.df.loc[start_date:end_date]

    def analyze_events(self):
        results = []
        for date_str, event_name in self.significant_events.items():
            event_date = pd.to_datetime(date_str)
            prices_around_event = self.get_prices_around_event(event_date, days_before=180, days_after=180)

            try:
                nearest_before_1m = self.df.index[self.df.index <= event_date - timedelta(days=30)][-1]
                nearest_after_1m = self.df.index[self.df.index >= event_date + timedelta(days=30)][0]
                price_before_1m = self.df.loc[nearest_before_1m, 'Price']
                price_after_1m = self.df.loc[nearest_after_1m, 'Price']
                change_1m = ((price_after_1m - price_before_1m) / price_before_1m) * 100
            except (IndexError, KeyError):
                change_1m = None
            
            try:
                nearest_before_3m = self.df.index[self.df.index <= event_date - timedelta(days=90)][-1]
                nearest_after_3m = self.df.index[self.df.index >= event_date + timedelta(days=90)][0]
                price_before_3m = self.df.loc[nearest_before_3m, 'Price']
                price_after_3m = self.df.loc[nearest_after_3m, 'Price']
                change_3m = ((price_after_3m - price_before_3m) / price_before_3m) * 100
            except (IndexError, KeyError):
                change_3m = None

            try:
                nearest_before_6m = self.df.index[self.df.index <= event_date - timedelta(days=180)][-1]
                nearest_after_6m = self.df.index[self.df.index >= event_date + timedelta(days=180)][0]
                price_before_6m = self.df.loc[nearest_before_6m, 'Price']
                price_after_6m = self.df.loc[nearest_after_6m, 'Price']
                change_6m = ((price_after_6m - price_before_6m) / price_before_6m) * 100
            except (IndexError, KeyError):
                change_6m = None
            
            if not prices_around_event.empty:
                try:
                    prices_before = prices_around_event.loc[:event_date]
                    prices_after = prices_around_event.loc[event_date:]
                    
                    cum_return_before = prices_before['Price'].pct_change().add(1).cumprod().iloc[-1] - 1
                    cum_return_after = prices_after['Price'].pct_change().add(1).cumprod().iloc[-1] - 1
                except:
                    cum_return_before = None
                    cum_return_after = None
            else:
                cum_return_before = None
                cum_return_after = None
            
            results.append({
                "Event": event_name,
                "Date": date_str,
                "Change_1M": change_1m,
                "Change_3M": change_3m,
                "Change_6M": change_6m,
                "Cumulative Return Before": cum_return_before,
                "Cumulative Return After": cum_return_after
            })

        event_impact_df = pd.DataFrame(results)
        logging.info("Event Impact Analysis: \n%s", event_impact_df)

        self.visualize_event_impact(event_impact_df)
    
    def visualize_event_impact(self, event_impact_df):
        # You can implement your visualization code here
        pass

# Example usage in your notebook
# import pandas as pd
# from analyzer import OilPriceAnalyzer

# Load your data
# data = pd.read_csv('your_data_file.csv')
# analyzer = OilPriceAnalyzer(data)
# analyzer.analyze_events()
