## 📝 The Data Analysis Workflow  

To analyze Brent oil prices and detect change points effectively, i will used the Suggested Approach:  

### **1️⃣ Data Collection & Understanding the Dataset**  
- Start by obtaining the **historical Brent oil prices** dataset (1987 - 2022).
- i Have alredy got this data **BrentOilPrices.csv**
-  
- The dataset  have:  
  - **Date**: The day oil prices were recorded.  
  - **Price**: Brent crude oil price (USD per barrel).  

### **2️⃣ Data Preprocessing** 🧹  
- Check for missing values and handle them appropriately.  
 
- Ensure data consistency (e.g., removing duplicate entries)if exist .  
  

### **3️⃣ Exploratory Data Analysis (EDA) 📊**  
- Plot the **price trends over time** to identify visible patterns.  
- Analyze statistical properties like **mean, variance, stationarity, seasonality, and volatility**.  
- Use **histograms, box plots, and rolling statistics** to get insights into price fluctuations.  

### **4️⃣ Change Point Detection & Statistical Modeling**  
- Apply **change point detection methods** to identify significant shifts in oil prices:  
  - **Bayesian Change Point Detection**  
  - **Likelihood Ratio Tests**  
  - **CUSUM (Cumulative Sum Control Chart)**  
  - **Pettitt’s Test**  
- Implement **time series models** to understand trends and volatility:  
  - **ARIMA (AutoRegressive Integrated Moving Average)**  
  - **GARCH (Generalized Autoregressive Conditional Heteroskedasticity)**  
  - **Bayesian Methods (PyMC3)** for probabilistic trend detection.  

### **5️⃣ Model Evaluation & Selection** ✅  
- Compare models using **evaluation metrics** like:  
  - **AIC/BIC** (Akaike and Bayesian Information Criteria)  
  - **RMSE (Root Mean Squared Error)**  
  - **MAPE (Mean Absolute Percentage Error)**  
- Select the best-performing model for predicting oil price fluctuations.  

### **6️⃣ Interpretation of Findings & Insights** 🔍  
- Correlate **change points** with real-world events (e.g., political, economic, regulatory).  
- Identify how external factors like **OPEC decisions, economic sanctions, and geopolitical events** impact oil prices.  

### **7️⃣ Communicating Results to Stakeholders** 📢  
- Determine the best way to **present insights**:  
  - **Interactive Dashboard (Flask + React)** to visualize trends and predictions.  
  - **Blog Report** summarizing key findings.also for for non-technical audiences 
  
