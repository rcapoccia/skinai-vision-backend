import os
import numpy as np
import cv2
import onnxruntime as ort
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq

app = Flask(__name__)
CORS(app)

# Configurazione Groq
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

MODEL_PATH = "skin_analyzer.onnx"
session = None
model_loaded = False
error_msg = ""

# Inizializzazione Sessione ONNX
try:
    if os.path.exists(MODEL_PATH):
        # NOTA: ONNX cercherà automaticamente skin_analyzer.onnx.data
        session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
        model_loaded = True
        print("✅ Modello caricato correttamente")
    else:
        error_msg = "File .onnx non trovato"
except Exception as e:
    error_msg = str(e)
    print(f"❌ Errore caricamento: {e}")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "online",
        "model_loaded": model_loaded,
        "error": error_msg,
        "onnx_file_present": os.path.exists(MODEL_PATH)
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    if not model_loaded:
        return jsonify({"error": f"Modello non pronto: {error_msg}"}), 500
    
    if 'image' not in request.files:
        return jsonify({"error": "Immagine mancante"}), 400
    
    try:
        file = request.files['image']
        # Pre-processing base
        nparr = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        # Inferenza
        input_name = session.get_inputs()[0].name
        raw_scores = session.run(None, {input_name: img})[0][0]

        results = {
            "rughe": round(float(raw_scores[0]) * 100, 1),
            "pori": round(float(raw_scores[1]) * 100, 1),
            "pigmentazione": round(float(raw_scores[2]) * 100, 1),
            "idratazione": round(float(raw_scores[3]) * 100, 1),
            "sensibilita": round(float(raw_scores[4]) * 100, 1)
        }

        return jsonify({"analysis": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Fondamentale per Railway: legge la porta assegnata
    port = int(os.environ.get("PORT", 3001))
    app.run(host='0.0.0.0', port=port)
