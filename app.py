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

# Configurazione CORS per comunicare con il frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- INIZIALIZZAZIONE ---
GROQ_CLIENT = None
if os.environ.get("GROQ_API_KEY"):
    GROQ_CLIENT = Groq(api_key=os.environ.get("GROQ_API_KEY"))

session = None
model_path = "skin_analyzer.onnx"
errore_caricamento = "Nessun errore"

# Caricamento del modello ONNX all'avvio
try:
    if os.path.exists(model_path):
        # Inizializza la sessione ONNX
        session = ort.InferenceSession(model_path)
        print("✅ Modello ONNX caricato con successo!")
    else:
        errore_caricamento = f"File {model_path} non trovato nel server."
except Exception as e:
    errore_caricamento = f"Errore tecnico ONNX: {str(e)}"
    print(f"❌ Errore critico nel caricamento del modello: {e}")

# --- ENDPOINT DI DIAGNOSTICA ---
@app.get("/health")
async def health():
    return {
        "status": "online",
        "model_loaded": session is not None,
        "onnx_file_present": os.path.exists(model_path),
        "groq_key_configured": os.environ.get("GROQ_API_KEY") is not None,
        "errore_dettagliato": errore_caricamento
    }

# --- ENDPOINT DI ANALISI ---
@app.post("/analyze")
async def analyze(file: UploadFile = File(...), questionnaire: str = Form("{}")):
    try:
        if session is None:
            return {"status": "error", "message": f"Il motore AI non è pronto: {errore_caricamento}"}

        # 1. Lettura Immagine
        img_data = await file.read()
        img = Image.open(io.BytesIO(img_data)).convert("RGB").resize((224, 224))
        
        # 2. Pre-processing per ONNX (Scala 0-1, formato CHW)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1)) # Da HWC a CHW
        arr = np.expand_dims(arr, 0)       # Aggiungi batch dimension
        
        # 3. Esecuzione Modello
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: arr})
        res = output[0][0] # Prendi i risultati del primo batch

        # 4. Calibrazione Scala 0-100 (Sincronizzato con il tuo Dashboard)
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

        # 5. Generazione Ragionamento con Groq
        ragionamento = "Analisi biometrica completata con successo."
        if GROQ_CLIENT:
            try:
                quiz_data = json.loads(questionnaire)
                prompt = f"Analisi pelle: {calibrated}. Contesto Utente: {quiz_data}. Spiega il risultato in 2 frasi semplici e professionali in italiano."
                chat = GROQ_CLIENT.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.1-8b-instant",
                )
                ragionamento = chat.choices[0].message.content
            except:
                pass # Fallback al messaggio standard se Groq fallisce

        return {
            "status": "success",
            "calibrated_scores": calibrated,
            "ragionamento": ragionamento
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    # Usa la porta fornita da Railway o la 8080 come default
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
