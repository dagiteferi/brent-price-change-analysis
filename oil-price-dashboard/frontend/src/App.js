import React, { useState, useEffect } from "react";
import axios from "axios";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

function App() {
  const [historicalData, setHistoricalData] = useState([]);
  const [eventsData, setEventsData] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [inputData, setInputData] = useState({
    GDP: 1.5,
    CPI: 2.1,
    Exchange_Rate: 0.75,
    Price_Pct_Change: 0.02,
    GDP_Pct_Change: 0.03,
    CPI_Pct_Change: 0.01,
    Exchange_Rate_Pct_Change: -0.01,
    Price_MA7: 68.5,
    Price_MA30: 69.0,
    Price_Volatility: 0.1,
  });

  // Fetch historical data
  const fetchHistoricalData = async () => {
    try {
      const response = await axios.get("http://127.0.0.1:5000/api/data");
      setHistoricalData(response.data);
    } catch (error) {
      console.error("Error fetching historical data:", error);
    }
  };

  // Fetch event analysis
  const fetchEventsData = async () => {
    try {
      const response = await axios.get("http://127.0.0.1:5000/api/events");
      setEventsData(response.data);
    } catch (error) {
      console.error("Error fetching events data:", error);
    }
  };

  // Fetch model metrics
  const fetchMetrics = async () => {
    try {
      const response = await axios.get("http://127.0.0.1:5000/api/metrics");
      setMetrics(response.data);
    } catch (error) {
      console.error("Error fetching metrics:", error);
    }
  };

  // Make a prediction
  const makePrediction = async () => {
    try {
      const response = await axios.post("http://127.0.0.1:5000/api/predict", inputData);
      setPrediction(response.data.predicted_oil_price);
    } catch (error) {
      console.error("Error making prediction:", error);
    }
  };

  // Handle input change
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setInputData({
      ...inputData,
      [name]: parseFloat(value),
    });
  };

  // Fetch data on component mount
  useEffect(() => {
    fetchHistoricalData();
    fetchEventsData();
    fetchMetrics();
  }, []);

  // Chart data for historical prices
  const chartData = {
    labels: historicalData.map((item) => item.Date),
    datasets: [
      {
        label: "Oil Price",
        data: historicalData.map((item) => item.Price),
        borderColor: "rgba(75, 192, 192, 1)",
        fill: false,
      },
    ],
  };

  return (
    <div className="App">
      <header>
        <h1>Oil Price Analysis Dashboard</h1>
      </header>

      <main>
        {/* Historical Data Section */}
        <section className="card">
          <h2>Historical Oil Prices</h2>
          <div className="container">
            <Line data={chartData} />
          </div>
        </section>

        {/* Event Analysis Section */}
        <section className="card">
          <h2>Event Analysis</h2>
          <div id="searchResults">
            <table>
              <thead>
                <tr>
                  <th>Event</th>
                  <th>Date</th>
                  <th>1M Change (%)</th>
                  <th>3M Change (%)</th>
                  <th>6M Change (%)</th>
                </tr>
              </thead>
              <tbody>
                {eventsData.map((event, index) => (
                  <tr key={index}>
                    <td>{event.Event}</td>
                    <td>{event.Date}</td>
                    <td>{event.Change_1M?.toFixed(2) ?? "N/A"}</td>
                    <td>{event.Change_3M?.toFixed(2) ?? "N/A"}</td>
                    <td>{event.Change_6M?.toFixed(2) ?? "N/A"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {/* Prediction Section */}
        <section className="card">
          <h2>Make a Prediction</h2>
          <form>
            {Object.keys(inputData).map((key) => (
              <div key={key}>
                <label>{key}:</label>
                <input
                  type="number"
                  name={key}
                  value={inputData[key]}
                  onChange={handleInputChange}
                />
              </div>
            ))}
            <button type="button" onClick={makePrediction}>
              Predict
            </button>
          </form>
          {prediction !== null && (
            <p>Predicted Oil Price: {prediction.toFixed(2)}</p>
          )}
        </section>

        {/* Metrics Section */}
        <section className="card">
          <h2>Model Performance Metrics</h2>
          {metrics ? (
            <ul>
              <li>RMSE: {metrics.RMSE?.toFixed(2)}</li>
              <li>MAE: {metrics.MAE?.toFixed(2)}</li>
              <li>R²: {metrics.R2?.toFixed(2)}</li>
            </ul>
          ) : (
            <p>Loading metrics...</p>
          )}
        </section>
      </main>

      <footer>
        <p>&copy; 2023 Oil Price Analysis Dashboard</p>
      </footer>
    </div>
  );
}

export default App;