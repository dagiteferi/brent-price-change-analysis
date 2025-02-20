import pandas as pd
import matplotlib.pyplot as plt
import logging
import os
import seaborn as sns
from datetime import timedelta
from scipy import stats

# Set up logging
log_file_path = 'logs/analysis.log'
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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
    start_date = event_date - timedelta(days=days_before)
    end_date = event_date + timedelta(days=days_after)
    return df.loc[start_date:end_date]

def analyze_events(df):
    results = []
    for date_str, event_name in significant_events.items():
        event_date = pd.to_datetime(date_str)
        prices_around_event = get_prices_around_event(df, event_date)

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

    visualize_event_impact(event_impact_df)

def visualize_event_impact(event_impact_df):
    # Implement your visualization code here
    pass

def visualize_event_impact(self, event_impact_df):
        plt.figure(figsize=(14, 8))
        for date_str, event_name in self.significant_events.items():
            event_date = pd.to_datetime(date_str)
            prices_around_event = self.get_prices_around_event(event_date, days_before=180, days_after=180)
            
            if not prices_around_event.empty:
                plt.plot(prices_around_event.index, prices_around_event['Price'], label=f"{event_name} ({date_str})")
                plt.axvline(event_date, color='red', linestyle='--', linewidth=0.8)
                plt.text(event_date, prices_around_event['Price'].max(), event_name, 
                         rotation=90, verticalalignment='bottom', fontsize=8)

        plt.title("Brent Oil Price Trends Around Key Events")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

        changes_data = event_impact_df.melt(id_vars=["Event", "Date"], 
                                              value_vars=["Change_1M", "Change_3M", "Change_6M"])
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Bar plot for percentage changes
        sns.barplot(data=changes_data, x="Event", y="value", hue="variable", ax=axes[0])
        axes[0].set_title("Percentage Change in Brent Oil Prices Before and After Events")
        axes[0].set_ylabel("Percentage Change")
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
        axes[0].legend(title="Change Period")

        returns_data = event_impact_df.melt(id_vars=["Event", "Date"], 
                                              value_vars=["Cumulative Return Before", "Cumulative Return After"])
        sns.barplot(data=returns_data, x="Event", y="value", hue="variable", ax=axes[1])
        axes[1].set_title("Cumulative Returns Before and After Events")
        axes[1].set_ylabel("Cumulative Return")
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
        axes[1].legend(title="Return Period")

        plt.tight_layout()
        plt.show()

        self.statistical_analysis()

def statistical_analysis(self):
        t_test_results = {}
        for date_str, event_name in self.significant_events.items():
            event_date = pd.to_datetime(date_str)
            prices_around = self.get_prices_around_event(event_date, days_before=180, days_after=180)
            
            if not prices_around.empty:
                before_prices = prices_around.loc[:event_date]['Price']
                after_prices = prices_around.loc[event_date:]['Price']
                
                if len(before_prices) > 1 and len(after_prices) > 1:
                    t_stat, p_val = stats.ttest_ind(before_prices, after_prices, nan_policy='omit')
                    t_test_results[event_name] = {"t-statistic": t_stat, "p-value": p_val}
                else:
                    t_test_results[event_name] = {"t-statistic": None, "p-value": None}

        t_test_df = pd.DataFrame(t_test_results).T
        print("\nT-Test Results:")
        print(t_test_df)


