from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io
import os
import onnxruntime as ort
import json
from groq import Groq

app = FastAPI(title="SkinGlow AI - Super Backend Cami")

# Configurazione CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inizializza Groq e ONNX
GROQ_CLIENT = Groq(api_key=os.environ.get("GROQ_API_KEY"))
session = None

try:
    model_path = "skin_analyzer.onnx"
    session = ort.InferenceSession(model_path)
    print(f"✅ Motore ONNX pronto")
except Exception as e:
    print(f"❌ Errore caricamento modello: {e}")

@app.post("/analyze")
async def analyze(file: UploadFile = File(...), questionnaire: str = Form("{}")):
    try:
        # 1. Lettura e Pre-processing Immagine
        img_data = await file.read()
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        img_resized = img.resize((224, 224))
        
        # Preparazione per ONNX (Scala 0-1, formato NCHW)
        arr = np.array(img_resized, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        arr = np.expand_dims(arr, 0)
        
        # 2. Inferenza Matematica (Numeri Puri)
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: arr})
        predictions = output[0][0] # Otteniamo i 6 parametri base
        
        # 3. Calibrazione e Scala 0-100
        # Trasformiamo i valori grezzi (spesso 0-1 o piccoli) in scala 0-100
        def to_100(val): return min(100, max(0, float(val) * 100))

        calibrated = {
            "rughe": to_100(predictions[0] * 0.8), # Correzione sensibilità
            "pori": to_100(predictions[1]),
            "macchie": to_100(predictions[2] * 0.9),
            "occhiaie": to_100(predictions[3]),
            "disidratazione": to_100(predictions[4]),
            "acne": to_100(predictions[5] * 0.7), # Riduzione falsi positivi acne
            "pelle_pulita_percent": to_100(1.0 - (predictions[5] * 0.5)) # Calcolo inverso
        }

        # 4. Generazione Ragionamento con Groq (Basato sui numeri reali)
        quiz_data = json.loads(questionnaire)
        prompt = f"""
        Analizza questi dati biometrici della pelle: {calibrated}.
        Contesto utente: {quiz_data}.
        Scrivi un breve ragionamento professionale (max 3 frasi) spiegando il risultato principale.
        Sii incoraggiante e preciso.
        """
        
        chat_completion = GROQ_CLIENT.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
        )
        ragionamento = chat_completion.choices[0].message.content

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
