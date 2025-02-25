import React, { useEffect, useState } from 'react';
import './App.css';

function App() {
    const [oilData, setOilData] = useState([]);
    const [eventData, setEventData] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        // Fetch historical oil price data
        fetch('/api/data')
            .then((response) => response.json())
            .then((data) => {
                setOilData(data);
                setLoading(false);
            })
            .catch((error) => {
                console.error("Error fetching oil data:", error);
            });

        // Fetch event analysis results
        fetch('/api/events')
            .then((response) => response.json())
            .then((data) => {
                setEventData(data);
            })
            .catch((error) => {
                console.error("Error fetching event data:", error);
            });
    }, []);

    if (loading) {
        return <div>Loading...</div>;
    }

    return (
        <div className="App">
            <h1>Brent Oil Price Dashboard</h1>
            <h2>Historical Oil Prices</h2>
            <table>
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Price</th>
                    </tr>
                </thead>
                <tbody>
                    {oilData.map((item) => (
                        <tr key={item.Date}>
                            <td>{item.Date}</td>
                            <td>{item.Price}</td>
                        </tr>
                    ))}
                </tbody>
            </table>

            <h2>Event Analysis</h2>
            <ul>
                {eventData.map((event) => (
                    <li key={event.Date}>
                        {event.Event} on {event.Date}: Change 1M: {event.Change_1M}%
                    </li>
                ))}
            </ul>
        </div>
    );
}

export default App;
