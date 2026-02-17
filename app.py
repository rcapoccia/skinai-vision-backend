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

# Percorso modello e inizializzazione
MODEL_PATH = "skin_analyzer.onnx"
session = None
model_loaded = False

try:
    if os.path.exists(MODEL_PATH):
        # Inizializza la sessione ONNX
        session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
        model_loaded = True
        print("✅ Modello ONNX caricato con successo.")
    else:
        print("❌ Errore: File skin_analyzer.onnx non trovato.")
except Exception as e:
    print(f"❌ Errore durante il caricamento del modello: {e}")

def process_image(image_bytes):
    # Decodifica immagine e resize per il modello (es. 224x224, adatta se il tuo modello usa altre size)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224)) 
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (20, 1)) # HWC to CHW
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "online",
        "model_loaded": model_loaded,
        "onnx_file_present": os.path.exists(MODEL_PATH),
        "groq_key_configured": bool(os.environ.get("GROQ_API_KEY"))
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    if not model_loaded:
        return jsonify({"error": "Modello non pronto"}), 500
    
    if 'image' not in request.files:
        return jsonify({"error": "Nessuna immagine fornita"}), 400
    
    file = request.files['image']
    img_data = process_image(file.read())
    
    # Esecuzione Inferenza
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_data})
    
    # MAPPING RISULTATI (Adatta i nomi in base al tuo modello specifico)
    # Assumiamo che il modello restituisca un array di probabilità/score
    raw_scores = outputs[0][0]
    
    # Calibrazione e normalizzazione 0-100
    results = {
        "rughe": round(float(raw_scores[0]) * 100, 2),
        "pori": round(float(raw_scores[1]) * 100, 2),
        "pigmentazione": round(float(raw_scores[2]) * 100, 2),
        "idratazione": round(float(raw_scores[3]) * 100, 2),
        "sensibilita": round(float(raw_scores[4]) * 100, 2)
    }

    # Analisi AI con Groq
    prompt = f"Analizza questi parametri cutanei (scala 0-100): {results}. Fornisci un consiglio professionale breve e tecnico per una routine skincare EUCLOTHA."
    
    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3-70b-8192",
    )
    
    return jsonify({
        "analysis": results,
        "reasoning": chat_completion.choices[0].message.content
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 3001))
    app.run(host='0.0.0.0', port=port)
