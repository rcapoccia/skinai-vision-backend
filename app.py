from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io
import os
import onnxruntime as ort
import json
from groq import Groq

app = FastAPI(title="SkinGlow AI - Super Backend")

# Configurazione CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Configurazione Groq
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_CLIENT = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# 2. Caricamento Modello ONNX
session = None
model_path = "skin_analyzer.onnx"

try:
    if os.path.exists(model_path):
        session = ort.InferenceSession(model_path)
        print("✅ MODELLO CARICATO: skin_analyzer.onnx trovato e attivo.")
    else:
        print(f"❌ ERRORE: Il file {model_path} non esiste nella cartella principale.")
except Exception as e:
    print(f"❌ ERRORE CRITICO: Impossibile avviare il motore ONNX: {e}")

# --- QUESTA È LA PARTE CHE MANCAVA ---
@app.get("/health")
async def health():
    return {
        "status": "online",
        "model_loaded": session is not None,
        "onnx_file_present": os.path.exists(model_path),
        "groq_key_configured": GROQ_API_KEY is not None
    }
# -------------------------------------

@app.post("/analyze")
async def analyze(file: UploadFile = File(...), questionnaire: str = Form("{}")):
    try:
        if session is None:
            return {"status": "error", "message": "Modello AI non caricato sul server."}

        # Lettura immagine
        img_data = await file.read()
        img = Image.open(io.BytesIO(img_data)).convert("RGB").resize((224, 224))
        
        # Pre-processing
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        arr = np.expand_dims(arr, 0)
        
        # Inferenza
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: arr})
        res = output[0][0] 

        # Fix scala 0-100
        def fix(v): return float(round(min(100, max(0, v * 100))))

        calibrated = {
            "rughe": fix(res[0]),
            "pori": fix(res[1]),
            "macchie": fix(res[2]),
            "occhiaie": fix(res[3]),
            "disidratazione": fix(res[4]),
            "acne": fix(res[5]),
            "pelle_pulita_percent": fix(1.0 - (res[5] * 0.5))
        }

        # Ragionamento Groq
        ragionamento = "Analisi biometrica completata."
        if GROQ_CLIENT:
            prompt = f"Analisi pelle: {calibrated}. Quiz: {questionnaire}. Spiega il risultato in 2 frasi in italiano."
            chat = GROQ_CLIENT.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
            )
            ragionamento = chat.choices[0].message.content

        return {
            "status": "success",
            "calibrated_scores": calibrated,
            "ragionamento": ragionamento
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
