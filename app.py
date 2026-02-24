import os
import base64
import json
import io
import hashlib
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

# Dimensione massima lato lungo per evitare timeout SSL su immagini grandi
MAX_IMAGE_SIZE = 1024


def prepare_image(image_bytes):
    """Ridimensiona se necessario, applica sharpening, restituisce base64 JPEG."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Resize se troppo grande (evita timeout SSL con PNG pesanti)
    w, h = img.size
    if max(w, h) > MAX_IMAGE_SIZE:
        ratio = MAX_IMAGE_SIZE / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

    # Dermoscope effect
    img = img.filter(ImageFilter.SHARPEN)

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def image_seed(image_bytes):
    """Seed deterministico dall'hash MD5 dell'immagine originale."""
    return int(hashlib.md5(image_bytes).hexdigest()[:8], 16)


PROMPT = """Sei un dermatologo AI esperto. Analizza attentamente l'immagine del viso fornita.

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
    "ragionamento": "<analisi dettagliata in italiano, 2-3 frasi>"
}

SCALA (100 = perfetto, 0 = problema grave):

RUGHE (visibilità delle rughe):
- 90-100: nessuna ruga visibile, pelle liscia
- 70-89: qualche linea di espressione fine, rughe minime
- 50-69: rughe moderate visibili sulla fronte o intorno agli occhi
- 30-49: rughe profonde e numerose
- 0-29: solchi profondi ovunque, pelle molto segnata

PORI (dimensione e visibilità dei pori):
- 90-100: pori invisibili
- 70-89: pori poco visibili
- 50-69: pori moderatamente visibili
- 0-49: pori molto dilatati e visibili

MACCHIE (uniformità del tono):
- 90-100: tono uniforme, nessuna macchia
- 70-89: minime discromie
- 50-69: alcune macchie o discromie moderate
- 0-49: macchie evidenti o iperpigmentazione diffusa

OCCHIAIE (cerchi scuri/gonfiore sotto gli occhi):
- 90-100: area sotto gli occhi luminosa, nessuna occhiaia
- 70-89: occhiaie lievi, appena percettibili
- 50-69: occhiaie moderate, visibili ma non marcate
- 30-49: occhiaie marcate, colorazione scura evidente
- 0-29: occhiaie molto scure o gonfiore pronunciato
ATTENZIONE: anche una leggera colorazione bluastra/violacea sotto gli occhi = punteggio 60-75 (non 90+)

DISIDRATAZIONE (livello di idratazione della pelle):
- 90-100: pelle luminosa, idratata, elastica
- 70-89: pelle ben idratata con piccole aree secche
- 50-69: pelle moderatamente disidratata, opaca in alcune zone
- 30-49: pelle chiaramente disidratata, opaca, poco elastica
- 0-29: pelle molto secca, opaca, con fine texture di disidratazione
ATTENZIONE: pelle opaca e poco elastica = punteggio 30-50, NON 60+

ACNE (presenza di imperfezioni/brufoli):
- 90-100: pelle pulita, nessuna imperfezione
- 70-89: rarissime imperfezioni
- 50-69: acne lieve
- 0-49: acne moderata o grave

PELLE_PULITA_PERCENT (salute generale complessiva):
- Media ponderata degli altri parametri

REGOLA FONDAMENTALE: la stessa immagine deve SEMPRE produrre gli stessi identici punteggi.
Basa i punteggi SOLO su ciò che vedi nell'immagine, non su stime demografiche."""


@app.get("/health")
async def health():
    return {"status": "online", "mode": "Maverick-Groq", "api_key": bool(os.environ.get("GROQ_API_KEY"))}


@app.post("/analyze")
@app.post("/analyze-dermoscope")
async def analyze(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        img_b64 = prepare_image(img_bytes)
        seed = image_seed(img_bytes)

        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": PROMPT
                        },
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                    ],
                }
            ],
            response_format={"type": "json_object"},
            temperature=0,
            seed=seed,
        )

        analysis_result = json.loads(completion.choices[0].message.content)
        required_fields = ["rughe", "pori", "macchie", "occhiaie", "disidratazione", "acne", "pelle_pulita_percent"]
        if "beauty_scores" not in analysis_result:
            analysis_result["beauty_scores"] = {}
        for field in required_fields:
            if field not in analysis_result["beauty_scores"]:
                analysis_result["beauty_scores"][field] = 50
        for field in required_fields:
            val = analysis_result["beauty_scores"][field]
            analysis_result["beauty_scores"][field] = max(0, min(100, int(val)))
        return analysis_result

    except Exception as e:
        return {"error": str(e), "status": "failed"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
