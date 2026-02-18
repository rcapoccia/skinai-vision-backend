import os
import base64
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from groq import Groq

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # Legge l'immagine e applica il dermoscopio con OpenCV
    img_bytes = await file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Effetto Maverick (Sharpening per rughe e pori)
    gaussian = cv2.GaussianBlur(img, (0, 0), 2.0)
    img = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
    
    _, buffer = cv2.imencode('.jpg', img)
    img_b64 = base64.b64encode(buffer).decode('utf-8')

    # Chiamata a Groq Vision (L'analisi la fanno loro!)
    completion = client.chat.completions.create(
        model="llama-3.2-11b-vision-preview",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "Analizza pelle: rughe, pori, acne, macchie. Rispondi solo in JSON."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
        ]}],
        response_format={"type": "json_object"}
    )
    return completion.choices[0].message.content
