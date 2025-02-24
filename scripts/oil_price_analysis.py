import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.preprocessing import StandardScaler

try:
    import seaborn as sns
except ImportError:
    print("Warning: Seaborn not found. Some plot styles may be affected.")
    sns = None

# Setup logging
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "oil_price_analysis.log")
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class OilPriceAnalyzer:
    def __init__(self, oil_data):
        """Initialize the OilPriceAnalyzer with oil price data."""
        self.oil_data = oil_data
        self.setup_plot_style()
        logging.info("OilPriceAnalyzer initialized.")
    
    def setup_plot_style(self):
        """Set up the plot style for visualization."""
        plt.style.use('default')
        if sns is not None:
            sns.set_palette("husl")
        else:
            logging.warning("Seaborn not available, using default colors.")
    
    def merge_and_clean_data(self, indicator_data, indicator_name):
        """Merge oil data with the indicator dataset and remove outliers."""
        logging.info(f"Merging and cleaning data for {indicator_name}.")
        merged_data = pd.merge(indicator_data, self.oil_data.reset_index(), on='Date')
        merged_data.dropna(inplace=True)
        
        # Remove outliers using IQR method
        Q1 = merged_data[indicator_name].quantile(0.25)
        Q3 = merged_data[indicator_name].quantile(0.75)
        IQR = Q3 - Q1
        merged_data = merged_data[
            (merged_data[indicator_name] >= Q1 - 1.5 * IQR) & 
            (merged_data[indicator_name] <= Q3 + 1.5 * IQR)
        ]
        logging.info(f"Data merged and cleaned for {indicator_name}.")
        return merged_data
    
    def calculate_statistics(self, merged_data, indicator_name):
        """Calculate statistical correlation between indicator and oil prices."""
        logging.info(f"Calculating statistics for {indicator_name}.")
        stats_dict = {}
        
        # Compute Pearson correlation
        correlation, p_value = stats.pearsonr(
            merged_data[indicator_name],
            merged_data['Price']
        )
        stats_dict['correlation'] = correlation
        stats_dict['p_value'] = p_value
        stats_dict['r_squared'] = correlation ** 2
        
        # Compute rolling correlation
        merged_data['rolling_corr'] = merged_data[indicator_name].rolling(
            window=180
        ).corr(merged_data['Price'])
        
        logging.info(f"Correlation: {correlation:.3f}, P-value: {p_value:.3e}, R-squared: {stats_dict['r_squared']:.3f}")
        return stats_dict, merged_data
    
    def create_visualization(self, merged_data, indicator_name, x_label, stats_dict):
        """Generate various visualizations to analyze the relationship."""
        logging.info(f"Creating visualization for {indicator_name}.")
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Analysis of {indicator_name} vs Oil Prices', fontsize=16)
        
        # Scatter plot with regression line
        axes[0, 0].scatter(merged_data[indicator_name], merged_data['Price'], alpha=0.5)
        z = np.polyfit(merged_data[indicator_name], merged_data['Price'], 1)
        p = np.poly1d(z)
        axes[0, 0].plot(merged_data[indicator_name], p(merged_data[indicator_name]), "r--")
        axes[0, 0].set_title('Scatter Plot with Regression Line')
        
        # Rolling correlation plot
        merged_data['rolling_corr'].plot(ax=axes[0, 1])
        axes[0, 1].set_title('6-Month Rolling Correlation')
        
        # Joint distribution plot
        axes[1, 0].hist2d(merged_data[indicator_name], merged_data['Price'], bins=50)
        axes[1, 0].set_title('Joint Distribution')
        
        # Time series comparison
        ax2 = axes[1, 1].twinx()
        merged_data[indicator_name].plot(ax=axes[1, 1], color='blue', label=indicator_name)
        merged_data['Price'].plot(ax=ax2, color='red', label='Oil Price')
        axes[1, 1].set_title('Time Series Comparison')
        
        plt.tight_layout()
        plt.show()
        logging.info(f"Visualization completed for {indicator_name}.")
    
    def analyze_granger_causality(self, merged_data, indicator_name, max_lag=12):
        """Perform Granger Causality test to check causal relationship."""
        logging.info(f"Performing Granger Causality test for {indicator_name}.")
        data = pd.DataFrame({
            'indicator': merged_data[indicator_name],
            'oil_price': merged_data['Price']
        })
        
        # Normalize data
        scaler = StandardScaler()
        data_scaled = pd.DataFrame(
            scaler.fit_transform(data),
            columns=data.columns,
            index=data.index
        )
        
        grangercausalitytests(data_scaled[['indicator', 'oil_price']], maxlag=max_lag, verbose=False)
        grangercausalitytests(data_scaled[['oil_price', 'indicator']], maxlag=max_lag, verbose=False)
        logging.info(f"Granger Causality analysis completed for {indicator_name}.")
    
    def analyze_indicator(self, indicator_data, indicator_name, x_label):
        """Analyze the relationship between an indicator and oil prices."""
        logging.info(f"Starting analysis for {indicator_name}.")
        merged_data = self.merge_and_clean_data(indicator_data, indicator_name)
        stats_dict, merged_data = self.calculate_statistics(merged_data, indicator_name)
        self.create_visualization(merged_data, indicator_name, x_label, stats_dict)
        self.analyze_granger_causality(merged_data, indicator_name)
        logging.info(f"Analysis completed for {indicator_name}.")

def analyze_indicators(gdp_data, inflation_data, unemployment_data, exchange_rate_data, oil_data):
    """Run analysis for multiple economic indicators."""
    logging.info("Starting analysis for all indicators.")
    analyzer = OilPriceAnalyzer(oil_data)
    
    analyzer.analyze_indicator(gdp_data, 'GDP', 'GDP (current US$)')
    analyzer.analyze_indicator(inflation_data, 'CPI', 'Inflation Rate (%)')
    analyzer.analyze_indicator(unemployment_data, 'Unemployment_Rate', 'Unemployment Rate (%)')
    analyzer.analyze_indicator(exchange_rate_data, 'Exchange_Rate', 'Exchange Rate (USD to Local Currency)')
    logging.info("All indicator analyses completed.")
