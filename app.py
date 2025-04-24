import os
import base64
import io
import joblib
import numpy as np
import pandas as pd
import requests
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI  # Dhenu's OpenAI-compatible client
import tensorflow as tf
from PIL import Image
from datetime import datetime
from deep_translator import GoogleTranslator
from gtts import gTTS
from gtts.lang import tts_langs
from indic_transliteration.sanscript import transliterate, DEVANAGARI, BENGALI, ORIYA, GUJARATI, KANNADA, MALAYALAM, GURMUKHI, TAMIL, TELUGU, ITRANS

load_dotenv()  # Load variables from .env file

app = Flask(__name__)

# -------------------------------
# Set Base & Model Directories
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
if not os.path.exists(MODEL_DIR):
    raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")

# -------------------------------
# Load Crop Recommendation Model and Scaler
# -------------------------------
try:
    crop_model_path = os.path.join(MODEL_DIR, "crop_rec.pkl")
    crop_model = joblib.load(crop_model_path)
    print("Loaded crop_model type:", type(crop_model))
    if not hasattr(crop_model, "predict"):
        raise ValueError("Loaded crop model does not have a predict method. Please re-save your model correctly.")
except Exception as e:
    print("Error loading crop model:", e)
    raise

try:
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    crop_scaler = joblib.load(scaler_path)
except Exception as e:
    print("Error loading crop scaler:", e)
    raise

# -------------------------------
# Load Fertilizer Recommendation Model and related files
# -------------------------------
try:
    fert_model_path = os.path.join(MODEL_DIR, "fert_rec.pkl")
    fert_model = joblib.load(fert_model_path)
except Exception as e:
    print("Error loading fertilizer model:", e)
    raise

try:
    fert_le_path = os.path.join(MODEL_DIR, "fert_le.pkl")
    fert_le = joblib.load(fert_le_path)
except Exception as e:
    print("Error loading fertilizer label encoder:", e)
    raise

try:
    fert_columns_path = os.path.join(MODEL_DIR, "fert_columns.pkl")
    fert_columns = joblib.load(fert_columns_path)
except Exception as e:
    print("Error loading fertilizer columns:", e)
    raise

# -------------------------------
# Load Plant Disease Detection Model (images resized to 224x224)
# -------------------------------
try:
    plant_disease_model_path = os.path.join(MODEL_DIR, "plant-disease.keras")
    disease_model = tf.keras.models.load_model(plant_disease_model_path)
except Exception as e:
    print("Error loading plant disease model:", e)
    raise

# -------------------------------
# Dhenu API configuration using the Dhenu model "dhenu2-in-8b-preview"
# -------------------------------
DHENU_API_KEY = os.getenv("DHENU_API_KEY")
if not DHENU_API_KEY:
    raise ValueError("DHENU_API_KEY environment variable is not set.")
client = OpenAI(base_url="https://api.dhenu.ai/v1", api_key=DHENU_API_KEY)

# -------------------------------
# Weather API key (provided)
# -------------------------------
WEATHER_API_KEY = "a7912fbcc99e1ff4f9f82b7cd56b7fec"

# -------------------------------
# Mapping for Indian language native scripts for transliteration
# -------------------------------
native_scripts = {
    "hi": DEVANAGARI,
    "bn": BENGALI,
    "or": ORIYA,       # Odia (as ORIYA in indic_transliteration)
    "gu": GUJARATI,
    "kn": KANNADA,
    "ml": MALAYALAM,
    "mr": DEVANAGARI,  # Marathi typically uses Devanagari
    "pa": GURMUKHI,
    "ta": TAMIL,
    "te": TELUGU
}

# --- Helper Functions for Chat Module ---

def translate_text(text, target_language, source_language="auto"):
    """
    Translate text using deep_translator's GoogleTranslator.
    """
    translator = GoogleTranslator(source=source_language, target=target_language)
    return translator.translate(text)

def synthesize_speech(text, language_code):
    """
    Convert text to speech using gTTS.
    If the desired language is not supported by gTTS, falls back to Hindi ("hi").
    Returns a base64-encoded MP3 string.
    """
    supported_languages = tts_langs()
    if language_code not in supported_languages:
        language_code = "hi"  # Fallback to Hindi
    tts = gTTS(text=text, lang=language_code)
    buffer = io.BytesIO()
    tts.write_to_fp(buffer)
    buffer.seek(0)
    audio_bytes = buffer.read()
    return base64.b64encode(audio_bytes).decode("utf-8")

# -------------------------------
# Routes
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    form_type = None  # "crop", "fertilizer", "weather", "irrigation", "chat"
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
                columns = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
                input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=columns)
                scaled_input = crop_scaler.transform(input_data)
                if hasattr(crop_model, "predict") and callable(crop_model.predict):
                    prediction = crop_model.predict(scaled_input)[0]
                else:
                    prediction = "Error: Loaded crop model does not have a predict method."
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
                data_encoded = pd.get_dummies(data, columns=["Soil Type", "Crop Type"], drop_first=True)
                data_encoded = data_encoded.reindex(columns=fert_columns, fill_value=0)
                pred_encoded = fert_model.predict(data_encoded)[0]
                prediction = fert_le.inverse_transform([pred_encoded])[0]
            elif form_type == "weather":
                # Weather Information branch
                city = request.form.get("city")
                if not city:
                    prediction = "No city provided."
                else:
                    current_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
                    current_resp = requests.get(current_url)
                    current_data = current_resp.json()
                    if current_data.get("cod") != 200:
                        prediction = current_data.get("message", "Error fetching weather data.")
                    else:
                        output = f"<h3>Weather for {city.capitalize()}</h3>"
                        main = current_data.get("main", {})
                        if "temp" in main:
                            output += f"<b>Temperature:</b> {main.get('temp', 'N/A')}°C<br>"
                        if "humidity" in main:
                            output += f"<b>Humidity:</b> {main.get('humidity', 'N/A')}%<br>"
                        if "pressure" in main:
                            output += f"<b>Pressure:</b> {main.get('pressure', 'N/A')} hPa<br>"
                        wind = current_data.get("wind", {})
                        if "speed" in wind:
                            output += f"<b>Wind Speed:</b> {wind.get('speed', 'N/A')} m/s<br>"
                        weather_list = current_data.get("weather", [])
                        if weather_list:
                            description = weather_list[0].get("description", "N/A").capitalize()
                            output += f"<b>Conditions:</b> {description}<br>"
                        prediction = output
            elif form_type == "irrigation":
                # Irrigation Schedule branch
                city = request.form.get("city")
                if not city:
                    prediction = "No city provided."
                else:
                    prediction = (
                        f"For {city.capitalize()}, it is recommended to irrigate every 2 days. "
                        "This schedule maintains optimal soil moisture and prevents overwatering."
                    )
        except Exception as e:
            prediction = f"Error: {str(e)}"
        return render_template("index.html", prediction=prediction, form_type=form_type)
    else:
        return render_template("index.html", prediction=prediction)

@app.route("/detect_disease", methods=["POST"])
def detect_disease():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['file']
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    try:
        image = Image.open(file.stream).convert("RGB")
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        preds = disease_model.predict(image_array)
        predicted_class = np.argmax(preds, axis=1)[0]
        disease_map = {
            0: "Apple_Apple_scab",
            1: "Apple_Black_rot",
            2: "Apple_Cedar_apple_rust",
            3: "Apple_healthy",
            4: "Blueberry_healthy",
            5: "Cherry_(including_sour)_Powdery_mildew",
            6: "Cherry_(including_sour)_healthy",
            7: "Corn_(maize)_Cercospora_leaf_spot_Gray_leaf_spot",
            8: "Corn_(maize)_Common_rust",
            9: "Corn_(maize)_Northern_Leaf_Blight",
            10: "Corn_(maize)_healthy",
            11: "Grape_Black_rot",
            12: "Grape_Esca_(Black_Measles)",
            13: "Grape_Leaf_blight_(Isariopsis_Leaf_Spot)",
            14: "Grape_healthy",
            15: "Orange_Haunglongbing_(Citrus_greening)",
            16: "Peach_Bacterial_spot",
            17: "Peach_healthy",
            18: "Pepper,_bell_Bacterial_spot",
            19: "Pepper,_bell_healthy",
            20: "Potato_Early_blight",
            21: "Potato_Late_blight",
            22: "Potato_healthy",
            23: "Raspberry_healthy",
            24: "Soybean_healthy",
            25: "Squash_Powdery_mildew",
            26: "Strawberry_Leaf_scorch",
            27: "Strawberry_healthy",
            28: "Tomato_Bacterial_spot",
            29: "Tomato_Early_blight",
            30: "Tomato_Yellow_Leaf_Curl_Virus",
            31: "Tomato_healthy",
            32: "Tomato_Late_blight",
            33: "Tomato_Leaf_Mold",
            34: "Tomato_Septoria_leaf_spot",
            35: "Tomato_Spider_mites_Two-spotted_spider_mite",
            36: "Tomato_mosaic_virus",
            37: "Tomato_Target_Spot"
        }
        result = disease_map.get(predicted_class, "Unknown Disease")
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Single Chat Endpoint ---
@app.route("/chat", methods=["POST"])
def chat_route():
    data = request.get_json()
    user_message = data.get("message")
    input_language = data.get("input_language", "en")
    output_language = data.get("output_language", "en")
    output_script = data.get("output_script", "native")  # "native" or "romanized"
    
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Translate user message to English if needed.
        input_text = translate_text(user_message, "en") if input_language != "en" else user_message

        # Send the (translated) message to Dhenu's API.
        stream = client.chat.completions.create(
            model="dhenu2-in-8b-preview",
            messages=[{"role": "user", "content": input_text}],
        )
        print("Dhenu API response:", stream)
        reply_english = None
        if isinstance(stream, dict):
            if "choices" in stream and stream["choices"]:
                first_choice = stream["choices"][0]
                if isinstance(first_choice, dict) and "message" in first_choice:
                    msg = first_choice["message"]
                    if isinstance(msg, dict) and "content" in msg:
                        reply_english = msg["content"]
            if not reply_english and "text" in stream:
                reply_english = stream["text"]
        else:
            if hasattr(stream, "choices") and stream.choices:
                first_choice = stream.choices[0]
                if hasattr(first_choice, "message") and hasattr(first_choice.message, "content"):
                    reply_english = first_choice.message.content

        if not reply_english:
            reply_english = "Sorry, I couldn't process your message."

        # Translate the reply to the desired output language if needed.
        reply_translated = translate_text(reply_english, output_language) if output_language != "en" else reply_english

        # If output should be romanized and the language is Indian, transliterate.
        if output_script == "romanized" and output_language in native_scripts:
            reply_translated = transliterate(reply_translated, native_scripts[output_language], ITRANS)

        # Synthesize speech using gTTS (fallback to Hindi if unsupported).
        audio_base64 = synthesize_speech(reply_translated, output_language)

        return jsonify({"reply": reply_translated, "audio": audio_base64})
    except Exception as e:
        print("Error in chat endpoint:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
