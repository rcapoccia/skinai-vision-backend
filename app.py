import os
import base64
import json
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
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
    """Processa l'immagine con OpenCV (Dermoscopio) per Groq."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Effetto dermoscopio (Unsharp Mask + Contrast)
    gaussian = cv2.GaussianBlur(img, (0, 0), 3)
    img = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

@app.get("/health")
async def health():
    return {"status": "online", "mode": "Maverick-Groq", "key_set": bool(os.environ.get("GROQ_API_KEY"))}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        # 1. Pre-processing Maverick
        img_b64 = dermoscope_effect(img_bytes)
        
        # 2. Chiamata a Groq Vision
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analizza questa immagine dermoscopica. Restituisci JSON con punteggi 0-100 per: rughe, pori, macchie, occhiaie, acne."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                    ],
                }
            ],
            model="llama-3.2-11b-vision-preview",
            response_format={"type": "json_object"}
        )
        
        return json.loads(chat_completion.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
