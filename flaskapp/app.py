from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("model/crop_rec.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    crop_prediction = None
    if request.method == "POST":
        try:
            # Get form data
            N = float(request.form["N"])
            P = float(request.form["P"])
            K = float(request.form["K"])
            temperature = float(request.form["temperature"])
            humidity = float(request.form["humidity"])
            ph = float(request.form["ph"])
            rainfall = float(request.form["rainfall"])

            # Scale input
            input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            scaled_input = scaler.transform(input_data)
            
            # Predict crop
            crop_prediction = model.predict(scaled_input)[0]
        except Exception as e:
            crop_prediction = f"Error: {str(e)}"
    
    return render_template("index.html", prediction=crop_prediction)

import os

port = int(os.environ.get("PORT", 10000))  # default to 10000 for local dev
app.run(host='0.0.0.0', port=port)
