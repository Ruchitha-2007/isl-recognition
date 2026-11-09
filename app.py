import os
import io
import json
import base64
import joblib
import numpy as np
import cv2
from collections import deque
from flask import Flask, render_template, request, jsonify

from isl_engine import ISLLandmarkExtractor
from word_suggester import WordSuggester

app = Flask(__name__, static_folder="static", template_folder="templates")

# Initialize Landmark Extractor
extractor = ISLLandmarkExtractor(min_detection_confidence=0.45, min_tracking_confidence=0.4)

# Initialize Word Suggester
suggester = WordSuggester()

# Paths to Dual Models & General Model
ALPHABETS_MODEL_PATH = "models/isl_alphabets_model.pkl"
ALPHABETS_ENCODER_PATH = "models/isl_alphabets_model_encoder.pkl"
WORDS_MODEL_PATH = "models/isl_words_model.pkl"
WORDS_ENCODER_PATH = "models/isl_words_model_encoder.pkl"
FALLBACK_MODEL_PATH = "models/isl_model.pkl"
FALLBACK_ENCODER_PATH = "models/label_encoder.pkl"

model_alphabets = None
encoder_alphabets = None
model_words = None
encoder_words = None
model_fallback = None
encoder_fallback = None

ACTIVE_MODE = "alphabets" # "alphabets" or "words"
PREDICTION_HISTORY = deque(maxlen=5)

CURRENT_WORD = ""
SENTENCE = []

def load_all_models():
    global model_alphabets, encoder_alphabets, model_words, encoder_words, model_fallback, encoder_fallback
    
    # 1. Alphabets Model
    if os.path.exists(ALPHABETS_MODEL_PATH) and os.path.exists(ALPHABETS_ENCODER_PATH):
        try:
            model_alphabets = joblib.load(ALPHABETS_MODEL_PATH)
            encoder_alphabets = joblib.load(ALPHABETS_ENCODER_PATH)
            print("Loaded dedicated Alphabets Model.")
        except Exception as e:
            print(f"Error loading alphabets model: {e}")
            
    # 2. Words & Phrases Model
    if os.path.exists(WORDS_MODEL_PATH) and os.path.exists(WORDS_ENCODER_PATH):
        try:
            model_words = joblib.load(WORDS_MODEL_PATH)
            encoder_words = joblib.load(WORDS_ENCODER_PATH)
            print("Loaded dedicated Words Model.")
        except Exception as e:
            print(f"Error loading words model: {e}")
            
    # 3. Fallback Model
    if os.path.exists(FALLBACK_MODEL_PATH) and os.path.exists(FALLBACK_ENCODER_PATH):
        try:
            model_fallback = joblib.load(FALLBACK_MODEL_PATH)
            encoder_fallback = joblib.load(FALLBACK_ENCODER_PATH)
        except Exception:
            pass

load_all_models()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/switch_mode", methods=["POST"])
def switch_mode():
    global ACTIVE_MODE, PREDICTION_HISTORY
    data = request.get_json(force=True)
    new_mode = data.get("mode", "alphabets").strip().lower()
    if new_mode in ("alphabets", "words"):
        ACTIVE_MODE = new_mode
        PREDICTION_HISTORY.clear()
    return jsonify({"status": "success", "mode": ACTIVE_MODE})

@app.route("/api/predict_frame", methods=["POST"])
def predict_frame():
    global ACTIVE_MODE, CURRENT_WORD, SENTENCE
    
    # Choose active model based on requested mode
    data = request.get_json(force=True)
    mode = data.get("mode", ACTIVE_MODE).strip().lower()
    if mode not in ("alphabets", "words"):
        mode = ACTIVE_MODE
        
    if mode == "words" and model_words is not None:
        active_model = model_words
        active_encoder = encoder_words
    elif mode == "alphabets" and model_alphabets is not None:
        active_model = model_alphabets
        active_encoder = encoder_alphabets
    else:
        # Fallback if specific model is still compiling
        active_model = model_alphabets or model_words or model_fallback
        active_encoder = encoder_alphabets or encoder_words or encoder_fallback
        
    if active_model is None or active_encoder is None:
        load_all_models()
        active_model = model_alphabets or model_words or model_fallback
        active_encoder = encoder_alphabets or encoder_words or encoder_fallback
        if active_model is None:
            return jsonify({"status": "error", "message": "Model not loaded"}), 400

    try:
        image_data = data.get("image", "")
        if "," in image_data:
            image_data = image_data.split(",")[1]
            
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"status": "error", "message": "Failed to decode frame"}), 400

        # Extract rotation-invariant dual hand landmarks
        features, annotated_frame, hands_detected = extractor.extract_landmarks(frame)
        
        predicted_sign = None
        confidence = 0.0
        
        if hands_detected > 0:
            features = features.reshape(1, -1)
            probs = active_model.predict_proba(features)[0]
            max_idx = np.argmax(probs)
            raw_confidence = float(probs[max_idx])
            raw_pred = active_encoder.inverse_transform([max_idx])[0]
            
            # Prediction stabilization with rolling window
            PREDICTION_HISTORY.append((raw_pred, raw_confidence))
            preds = [p[0] for p in PREDICTION_HISTORY]
            counts = {p: preds.count(p) for p in set(preds)}
            most_frequent = max(counts, key=counts.get)
            
            # Responsive threshold (3 out of 5 frames, confidence >= 0.25)
            if counts[most_frequent] >= 3 and raw_confidence >= 0.25:
                predicted_sign = most_frequent
                confidence = raw_confidence
        else:
            PREDICTION_HISTORY.clear()
            
        # Base64 encode annotated frame
        _, buffer = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")
        
        payload = get_text_payload()
        
        return jsonify({
            "status": "success",
            "mode": mode,
            "hands_detected": hands_detected,
            "predicted_sign": predicted_sign,
            "confidence": round(confidence * 100, 1) if predicted_sign else 0.0,
            "annotated_image": annotated_b64,
            "current_word": payload["current_word"],
            "sentence": payload["sentence"],
            "full_text": payload["full_text"],
            "suggestions": payload["suggestions"]
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

def get_text_payload():
    full_sentence = " ".join(SENTENCE)
    if full_sentence and CURRENT_WORD:
        full_text = f"{full_sentence} {CURRENT_WORD}"
    elif CURRENT_WORD:
        full_text = CURRENT_WORD
    elif full_sentence:
        full_text = f"{full_sentence} "
    else:
        full_text = ""
    suggestions = suggester.suggest(CURRENT_WORD) if CURRENT_WORD else []
    return {
        "current_word": CURRENT_WORD,
        "sentence": full_sentence,
        "full_text": full_text,
        "suggestions": suggestions
    }

@app.route("/api/append_character", methods=["POST"])
def append_character():
    global CURRENT_WORD, SENTENCE
    data = request.get_json(force=True)
    char = data.get("char", "").strip().upper()
    
    if char == "SPACE":
        if CURRENT_WORD:
            SENTENCE.append(CURRENT_WORD)
            CURRENT_WORD = ""
    elif char == "BACKSPACE":
        if CURRENT_WORD:
            CURRENT_WORD = CURRENT_WORD[:-1]
        elif SENTENCE:
            CURRENT_WORD = SENTENCE.pop()
    elif len(char) == 1 and (char.isalpha() or char.isdigit()):
        CURRENT_WORD += char
    elif len(char) > 1 and char not in ("SPACE", "BACKSPACE"):
        # Full word sign (e.g. HELLO, HOW ARE YOU, THANK YOU)
        if CURRENT_WORD:
            SENTENCE.append(CURRENT_WORD)
            CURRENT_WORD = ""
        SENTENCE.append(char)
        
    return jsonify(get_text_payload())

@app.route("/api/accept_suggestion", methods=["POST"])
def accept_suggestion():
    global CURRENT_WORD, SENTENCE
    data = request.get_json(force=True)
    word = data.get("word", "").strip().upper()
    
    if word:
        SENTENCE.append(word)
        CURRENT_WORD = ""
        
    return jsonify(get_text_payload())

@app.route("/api/clear_text", methods=["POST"])
def clear_text():
    global CURRENT_WORD, SENTENCE
    CURRENT_WORD = ""
    SENTENCE = []
    return jsonify(get_text_payload())

@app.route("/api/suggest", methods=["GET"])
def get_suggestions():
    query = request.args.get("query", "").strip()
    suggestions = suggester.suggest(query)
    return jsonify({"query": query, "suggestions": suggestions})

@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    mode = request.args.get("mode", ACTIVE_MODE).strip().lower()
    if mode == "words":
        path = "models/words_evaluation_report.json"
    else:
        path = "models/alphabets_evaluation_report.json"
        
    if not os.path.exists(path):
        path = "models/evaluation_report.json"
        
    if os.path.exists(path):
        with open(path, "r") as f:
            return jsonify(json.load(f))
            
    return jsonify({"status": "no_metrics"})

@app.route("/api/signs_list", methods=["GET"])
def get_signs_list():
    mode = request.args.get("mode", ACTIVE_MODE).strip().lower()
    if mode == "words" and encoder_words is not None:
        signs = list(encoder_words.classes_)
    elif encoder_alphabets is not None:
        signs = list(encoder_alphabets.classes_)
    elif encoder_fallback is not None:
        signs = list(encoder_fallback.classes_)
    else:
        signs = [chr(i) for i in range(ord('A'), ord('Z')+1)]
    return jsonify({"signs": signs, "mode": mode})

if __name__ == "__main__":
    print("Starting Indian Sign Language Web Platform on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
