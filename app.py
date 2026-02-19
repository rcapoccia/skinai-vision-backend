'''
import os
import base64
import json
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq

app = FastAPI(title="SkinGlow AI - Maverick-Groq")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def dermoscope_effect(image_bytes):
    """Simula l'effetto dermoscopio per esaltare i dettagli della pelle."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gaussian = cv2.GaussianBlur(img, (0, 0), 3)
    img = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

@app.get("/health")
async def health():
    return {"status": "online", "mode": "Maverick-Groq", "api_key": bool(os.environ.get("GROQ_API_KEY"))}

# Unifichiamo entrambi gli endpoint richiesti dal frontend in un'unica funzione
@app.post("/analyze")
@app.post("/analyze-dermoscope")
async def analyze(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        img_b64 = dermoscope_effect(img_bytes)
        
        # Prompt aggiornato: richiede la struttura JSON esatta che il frontend si aspetta
        completion = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": """Sei un dermatologo AI. Analizza l'immagine della pelle fornita.
                            Restituisci ESCLUSIVAMENTE un oggetto JSON con la seguente struttura esatta, senza testo o spiegazioni aggiuntive:
                            {
                                "beauty_scores": {
                                    "rughe": <punteggio 0-100>,
                                    "pori": <punteggio 0-100>,
                                    "macchie": <punteggio 0-100>,
                                    "occhiaie": <punteggio 0-100>,
                                    "disidratazione": <punteggio 0-100>,
                                    "acne": <punteggio 0-100>,
                                    "pelle_pulita_percent": <punteggio 0-100>
                                },
                                "ragionamento": "<breve analisi testuale in italiano dei risultati>"
                            }
                            La scala è 0-100, dove 100 indica una pelle perfetta e 0 un problema grave."""
                        },
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                    ],
                }
            ],
            response_format={"type": "json_object"}
        )
        
        analysis_result = json.loads(completion.choices[0].message.content)
        
        # Fallback di sicurezza: garantisce che la struttura sia sempre valida per il frontend
        required_fields = ["rughe", "pori", "macchie", "occhiaie", "disidratazione", "acne", "pelle_pulita_percent"]
        if "beauty_scores" not in analysis_result:
            analysis_result["beauty_scores"] = {}
        
        for field in required_fields:
            if field not in analysis_result["beauty_scores"]:
                analysis_result["beauty_scores"][field] = 50 # Valore neutro di default

        return analysis_result

    except Exception as e:
        return {"error": str(e), "status": "failed"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
'''
