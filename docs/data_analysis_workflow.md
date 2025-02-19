# 📝 The Data Analysis Workflow  

To analyze Brent oil prices and detect change points effectively, I used the following approach:
- **Objective**:  
  - Identify significant change points in Brent oil prices over time.  
  - Forecast future oil price trends and volatility.  
  - Correlate price changes with real-world events.  

---

## **1️⃣ Data Collection & Understanding the Dataset**  
- The dataset used is **BrentOilPrices.csv** (1987 - 2022).  
- It contains:  
  - **Date**: The day oil prices were recorded.  
  - **Price**: Brent crude oil price (USD per barrel).  
 
---

### **2️⃣ Data Preprocessing** 🧹  
- Check for missing values and handle them appropriately.  
- Ensure data consistency (e.g., remove duplicate entries if they exist).  

---

### **3️⃣ Exploratory Data Analysis (EDA) 📊**  
- Plot **price trends over time** to identify visible patterns.  
- Analyze statistical properties such as **mean, variance, stationarity, seasonality, and volatility**.  
- Use **histograms, box plots, and rolling statistics** to gain insights into price fluctuations.  

---

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

---

## **5️⃣ Model Evaluation & Selection** ✅  
- Compare models using **evaluation metrics** such as:  
  - **AIC/BIC** (Akaike and Bayesian Information Criteria)  
  - **RMSE (Root Mean Squared Error)**  
  - **MAPE (Mean Absolute Percentage Error)**  
- Perform **cross-validation** to assess model robustness.  
- Select the best-performing model for predicting oil price fluctuations.   

---

### **6️⃣ Interpretation of Findings & Insights** 🔍  
- Correlate **change points** with real-world events (e.g., political, economic, and regulatory changes).  
- Identify how external factors such as **OPEC decisions, economic sanctions, and geopolitical events** impact oil prices.  

---

## **7️⃣ Communicating Results to Stakeholders** 📢  
- Present insights through:  
  - **Interactive Dashboard (Flask + React)**: Visualize trends, change points, and forecasts.  
  - **Blog Report**: Summarize key findings, making it accessible to both technical and non-technical audiences. 

---


# 📊 Understanding the Model and Data  

To effectively analyze Brent oil prices, I will use various time series models and understand how the data is generated. This section outlines the approach I will take.  

---

## 📚 1. Reviewing Key References  
Before implementing any models, I will:  
- Read **research papers and books** on time series forecasting.  

---

## **2️⃣ Choosing Suitable Models for Time Series Analysis**  

I will use the following models to analyze oil price fluctuations:  

### **🔹 ARIMA (AutoRegressive Integrated Moving Average)**  
- **Why?** ARIMA is useful for forecasting price trends based on historical patterns.  
- **How?** I will fit an ARIMA model and evaluate its ability to predict price movements.  
- **Inputs**: Historical price data.  
- **Parameters**: (p, d, q) for autoregressive, differencing, and moving average components.  
- **Outputs**: Forecasted prices and confidence intervals.  

### **🔹 GARCH (Generalized Autoregressive Conditional Heteroskedasticity)**  
- **Why?** Oil prices exhibit volatility, and GARCH models this volatility effectively.  
- **How?** I will use GARCH to measure and predict price fluctuations over time.  
- **Inputs**: Historical price data.  
- **Parameters**: (p, q) for volatility modeling.  
- **Outputs**: Volatility forecasts.  

### **🔹 Bayesian Change Point Detection**  
- **Why?** This method helps identify significant shifts in price trends due to external events.  
- **How?** I will implement Bayesian models using PyMC3 to detect these change points.  
- **Inputs**: Historical price data.  
- **Parameters**: Prior distributions for change points.  
- **Outputs**: Detected change points and their probabilities.  
 

---

## 🔍 3. Understanding How These Models Represent Data  

Brent oil prices are influenced by factors like **supply-demand dynamics, OPEC decisions, global crises, and market speculation**.  
To model these factors:  
- **ARIMA will help capture price trends and seasonality.**  
- **GARCH will analyze price volatility.**  
- **Bayesian methods will detect structural breaks in price movements.**  


---

## **4️⃣ Expected Outputs & Limitations**  

### **📊 Expected Outputs**  
By applying these models, I aim to:  
- Forecast **future Brent oil prices** based on historical data.  
- Identify **key change points** in price movements.  
- Assess **volatility levels** to understand market risk.  

### **⚠️ Limitations to Consider**  
- **ARIMA assumes stationarity**, but oil prices are affected by unexpected global events.  
- **GARCH models price volatility**, but requires careful parameter tuning.  
- **Change point detection methods depend on assumptions** that might not always hold.  
- **Macroeconomic and political factors are unpredictable**, impacting the accuracy of forecasts.    

---

By following this workflow, I aim to deliver actionable insights into Brent oil price trends and volatility, while effectively communicating results to stakeholders.

