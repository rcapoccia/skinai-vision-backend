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

# Il motore è Groq, non serve ONNX
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def dermoscope_effect(image_bytes):
    """Simula dermoscopio per esaltare rughe e pori tramite OpenCV."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Sharpening per rendere visibili i dettagli sottocutanei
    gaussian = cv2.GaussianBlur(img, (0, 0), 3)
    img = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

@app.get("/health")
async def health():
    return {"status": "online", "mode": "Maverick-Groq", "api_key": bool(os.environ.get("GROQ_API_KEY"))}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        img_b64 = dermoscope_effect(img_bytes)
        
        # L'analisi la fa Groq Vision (Zero carico su Railway)
        completion = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analizza la pelle in questa immagine dermoscopica. Restituisci un JSON con punteggi 0-100 per: rughe, pori, macchie, acne."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                    ],
                }
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
