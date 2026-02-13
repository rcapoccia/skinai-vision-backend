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

# --- INIZIALIZZAZIONE ---
GROQ_CLIENT = None
if os.environ.get("GROQ_API_KEY"):
    GROQ_CLIENT = Groq(api_key=os.environ.get("GROQ_API_KEY"))

session = None
model_path = "skin_analyzer.onnx"
debug_error = "Nessun errore"

# Tentativo di caricamento modello
try:
    if os.path.exists(model_path):
        # Carichiamo il modello ONNX
        session = ort.InferenceSession(model_path)
        print("✅ Modello caricato con successo!")
    else:
        debug_error = f"File {model_path} non trovato nel server"
except Exception as e:
    debug_error = f"Errore ONNX: {str(e)}"
    print(f"❌ Errore caricamento: {e}")

# --- ENDPOINT DI CONTROLLO ---
@app.get("/health")
async def health():
    return {
        "status": "online",
        "model_loaded": session is not None,
        "onnx_file_present": os.path.exists(model_path),
        "groq_key_configured": os.environ.get("GROQ_API_KEY") is not None,
        "errore_tecnico": debug_error
    }

# --- ENDPOINT DI ANALISI ---
@app.post("/analyze")
async def analyze(file: UploadFile = File(...), questionnaire: str = Form("{}")):
    try:
        if session is None:
            return {"status": "error", "message": f"IA non pronta: {debug_error}"}

        # 1. Lettura Immagine
        img_data = await file.read()
        img = Image.open(io.BytesIO(img_data)).convert("RGB").resize((224, 224))
        
        # 2. Pre-processing (Modello ONNX vuole float32 e formato CHW)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        arr = np.expand_dims(arr, 0)
        
        # 3. Inferenza
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: arr})
        res = output[0][0] 

        # 4. Fix Scala 0-100 (Sincronizzato con Frontend)
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

        # 5. Ragionamento AI con Groq
        ragionamento = "Analisi biometrica completata."
        if GROQ_CLIENT:
            try:
                quiz_data = json.loads(questionnaire)
                prompt = f"Analisi pelle: {calibrated}. Contesto: {quiz_data}. Commenta in 2 frasi in italiano."
                chat = GROQ_CLIENT.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.1-8b-instant",
                )
                ragionamento = chat.choices[0].message.content
            except:
                pass

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
