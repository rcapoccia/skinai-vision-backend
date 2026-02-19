import os
import base64
import json
import io
from PIL import Image, ImageFilter
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
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    sharpened = img.filter(ImageFilter.SHARPEN)
    buffer = io.BytesIO()
    sharpened.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

@app.get("/health")
async def health():
    return {"status": "online", "mode": "Maverick-Groq", "api_key": bool(os.environ.get("GROQ_API_KEY"))}

@app.post("/analyze")
@app.post("/analyze-dermoscope")
async def analyze(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        img_b64 = dermoscope_effect(img_bytes)

        completion = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Sei un dermatologo AI. Analizza l'immagine della pelle fornita.
Restituisci ESCLUSIVAMENTE un oggetto JSON con questa struttura, senza testo aggiuntivo:
{
    "beauty_scores": {
        "rughe": <0-100>,
        "pori": <0-100>,
        "macchie": <0-100>,
        "occhiaie": <0-100>,
        "disidratazione": <0-100>,
        "acne": <0-100>,
        "pelle_pulita_percent": <0-100>
    },
    "ragionamento": "<analisi breve in italiano>"
}
Scala: 100 = pelle perfetta, 0 = problema grave."""
                        },
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                    ],
                }
            ],
            response_format={"type": "json_object"}
        )

        analysis_result = json.loads(completion.choices[0].message.content)

        required_fields = ["rughe", "pori", "macchie", "occhiaie", "disidratazione", "acne", "pelle_pulita_percent"]
        if "beauty_scores" not in analysis_result:
            analysis_result["beauty_scores"] = {}
        for field in required_fields:
            if field not in analysis_result["beauty_scores"]:
                analysis_result["beauty_scores"][field] = 50

        return analysis_result

    except Exception as e:
        return {"error": str(e), "status": "failed"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
