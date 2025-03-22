import os
import joblib
import numpy as np
import pandas as pd
import requests
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI  # Dhenu provides an OpenAI-compatible client

load_dotenv()  # Load environment variables from .env file

app = Flask(__name__)

# Load Crop model and scaler
crop_model = joblib.load("model/crop_rec.pkl")
crop_scaler = joblib.load("model/scaler.pkl")

# Load Fertilizer model, label encoder, and training columns list
fert_model = joblib.load("model/fert_rec.pkl")
fert_le = joblib.load("model/fert_le.pkl")
fert_columns = joblib.load("model/fert_columns.pkl")  # List of feature names after one-hot encoding

# Dhenu API configuration using the Dhenu model "dhenu2-in-8b-preview"
DHENU_API_KEY = os.getenv("DHENU_API_KEY")
client = OpenAI(base_url="https://api.dhenu.ai/v1", api_key=DHENU_API_KEY)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    form_type = None  # Indicates which form was submitted (crop or fertilizer)
    if request.method == "POST":
        form_type = request.form.get("form_type")
        try:
            if form_type == "crop":
                # Crop Recommendation branch
                N = float(request.form["N"])
                P = float(request.form["P"])
                K = float(request.form["K"])
                temperature = float(request.form["temperature"])
                humidity = float(request.form["humidity"])
                ph = float(request.form["ph"])
                rainfall = float(request.form["rainfall"])
                input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
                scaled_input = crop_scaler.transform(input_data)
                prediction = crop_model.predict(scaled_input)[0]
            elif form_type == "fertilizer":
                # Fertilizer Recommendation branch
                temperature = float(request.form["temperature_fert"])
                humidity = float(request.form["humidity_fert"])
                moisture = float(request.form["moisture"])
                soil_type = request.form["soil_type"]
                crop_type = request.form["crop_type"]
                nitrogen = float(request.form["nitrogen"])
                potassium = float(request.form["potassium"])
                phosphorous = float(request.form["phosphorous"])
                
                # Create DataFrame with proper column names matching training data
                data = pd.DataFrame({
                    "Temperature": [temperature],
                    "Humidity": [humidity],
                    "Moisture": [moisture],
                    "Soil Type": [soil_type],
                    "Crop Type": [crop_type],
                    "Nitrogen": [nitrogen],
                    "Potassium": [potassium],
                    "Phosphorous": [phosphorous]
                })
                
                # One-hot encode the categorical features and reindex using saved training columns
                data_encoded = pd.get_dummies(data, columns=["Soil Type", "Crop Type"], drop_first=True)
                data_encoded = data_encoded.reindex(columns=fert_columns, fill_value=0)
                
                # Predict with fertilizer model and decode the label
                pred_encoded = fert_model.predict(data_encoded)[0]
                prediction = fert_le.inverse_transform([pred_encoded])[0]
        except Exception as e:
            prediction = f"Error: {str(e)}"
    
    return render_template("index.html", prediction=prediction, form_type=form_type)

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Send the chat message to Dhenu's API
        stream = client.chat.completions.create(
            model="dhenu2-in-8b-preview",
            messages=[{"role": "user", "content": user_message}],
        )
        # Debug log the raw response for inspection
        print("Dhenu API response:", stream)

        # Initialize reply variable
        reply = None
        # If the response is a dictionary, attempt extraction
        if isinstance(stream, dict):
            if "choices" in stream and isinstance(stream["choices"], list) and len(stream["choices"]) > 0:
                first_choice = stream["choices"][0]
                if isinstance(first_choice, dict) and "message" in first_choice:
                    msg = first_choice["message"]
                    if isinstance(msg, dict) and "content" in msg:
                        reply = msg["content"]
            if not reply and "text" in stream:
                reply = stream["text"]
        else:
            # Otherwise, try attribute access (if stream is an object)
            if hasattr(stream, "choices") and len(stream.choices) > 0:
                first_choice = stream.choices[0]
                if hasattr(first_choice, "message") and hasattr(first_choice.message, "content"):
                    reply = first_choice.message.content

        if not reply:
            reply = "Sorry, I couldn't process your message."
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
