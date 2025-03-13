import requests
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

# ---- Step 1: Flask Setup ----
app = Flask(__name__)

# ---- Step 2: Fetch Live Weather Data ----
API_KEY = "d52921978cbda8522b864c5ca7ab42b8"  # 🔹 Replace with your OpenWeatherMap API Key
BASE_URL = "http://api.openweathermap.org/data/2.5/weather?"

def get_weather_data(city):
    """Fetch live weather data from OpenWeatherMap API."""
    url = f"{BASE_URL}q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        weather_info = {
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],
            "wind_speed": data["wind"]["speed"],
            "weather": data["weather"][0]["main"]
        }
        return weather_info
    else:
        return None

# ---- Step 3: Train a Simple ML Model ----
past_weather = pd.DataFrame({
    "temperature": np.random.uniform(20, 35, 100),
    "humidity": np.random.uniform(50, 80, 100),
    "pressure": np.random.uniform(1000, 1025, 100),
    "wind_speed": np.random.uniform(0, 10, 100)
})

# Target variable: Predict next day's temperature
past_weather["target_temp"] = past_weather["temperature"].shift(-1)
past_weather.dropna(inplace=True)

# Train the model
X = past_weather.drop("target_temp", axis=1)
y = past_weather["target_temp"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# ---- Step 4: Flask Routes ----
@app.route("/", methods=["GET", "POST"])
def home():
    print("Request received")  # Debugging print

    prediction = None
    if request.method == "POST":
        city = request.form.get("city")
        date = request.form.get("date")
        time = request.form.get("time")

        print(f"User Input: City={city}, Date={date}, Time={time}")  # Debugging print

        weather_data = get_weather_data(city)
        
        if weather_data:
            print("Weather data received:", weather_data)  # Debugging print

            # Convert live data to model input
            input_data = pd.DataFrame([[
                weather_data["temperature"],
                weather_data["humidity"],
                weather_data["pressure"],
                weather_data["wind_speed"]
            ]], columns=["temperature", "humidity", "pressure", "wind_speed"])
            
            input_data_scaled = scaler.transform(input_data)
            predicted_temp = model.predict(input_data_scaled)[0]

            prediction = {
                "city": city,
                "date": date,
                "time": time,
                "current_temperature": weather_data["temperature"],
                "predicted_temperature": round(predicted_temp, 2),
                "humidity": weather_data["humidity"],
                "pressure": weather_data["pressure"],
                "wind_speed": weather_data["wind_speed"],
                "weather_condition": weather_data["weather"]
            }

            print("Prediction:", prediction)  # Debugging print

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
