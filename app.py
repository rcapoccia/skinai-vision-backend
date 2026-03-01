import os
import base64
import json
import io
import hashlib
import numpy as np
import cv2
from PIL import Image, ImageFilter
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq

app = FastAPI(title="SkinGlow AI - Maverick-Groq-OpenCV")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Haar Cascade for face detection (bundled with OpenCV, no extra deps)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

MAX_IMAGE_SIZE = 1024

def prepare_image_opencv(image_bytes):
    # 1. Decode to OpenCV format (RGB)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    # 2. Face detection and crop with padding
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        ih, iw = img_cv.shape[:2]
        pad_w = int(w * 0.25)
        pad_h = int(h * 0.25)
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(iw, x + w + pad_w)
        y2 = min(ih, y + h + pad_h)
        img_cv = img_cv[y1:y2, x1:x2]

    # 3. CLAHE for uniform illumination (applied on L channel of LAB)
    lab = cv2.cvtColor(img_cv, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    img_cv = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)

    # 4. Gamma correction (brighten slightly for better detail visibility)
    gamma = 1.2
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype="uint8")
    img_cv = cv2.LUT(img_cv, table)

    # 5. Convert to PIL, resize to max 1024px, sharpen, encode to JPEG
    img_pil = Image.fromarray(img_cv)
    w, h = img_pil.size
    if max(w, h) > MAX_IMAGE_SIZE:
        ratio = MAX_IMAGE_SIZE / max(w, h)
        img_pil = img_pil.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    img_pil = img_pil.filter(ImageFilter.SHARPEN)

    buffer = io.BytesIO()
    img_pil.save(buffer, format="JPEG", quality=85)
    processed_bytes = buffer.getvalue()

    return base64.b64encode(processed_bytes).decode("utf-8"), processed_bytes

def image_seed(image_bytes):
    return int(hashlib.md5(image_bytes).hexdigest()[:8], 16)

PROMPT_A = """Sei un assistente dermatologico che valuta SOLO la TEXTURE della pelle del viso
(rughe, pori, acne) osservando una foto frontale ben illuminata.

Devi restituire ESCLUSIVAMENTE un JSON con 3 campi numerici (0-100):
{
  "rughe": ...,
  "pori": ...,
  "acne": ...
}

Linee guida IMPORTANTI:
- Valuta ogni parametro INDIPENDENTEMENTE, come se 3 specialisti diversi non si parlassero.
- NON considerare colore, macchie pigmentarie, occhiaie o luminosità globale.
- Immagina di avere solo una mappa in scala di grigi della texture del viso.

Significato della scala (0-100, dove 100 = situazione migliore possibile):
- Rughe:
  90-100 = pelle liscia, nessuna ruga visibile nemmeno in espressione
  75-89  = leggere linee di espressione, rughe minime
  60-74  = rughe visibili nelle aree tipiche (fronte, contorno occhi, naso-labbiale)
  40-59  = rughe marcate, più aree coinvolte
  0-39   = rughe profonde e diffuse

- Pori:
  90-100 = pori quasi invisibili anche da vicino
  75-89  = pori piccoli, visibili solo da vicino
  60-74  = pori moderatamente visibili su naso e guance
  40-59  = pori dilatati evidenti su più zone
  0-39   = pori molto dilatati, texture irregolare

- Acne (brufoli/lesioni attive):
  90-100 = nessuna lesione attiva
  75-89  = poche lesioni isolate
  60-74  = acne lieve-moderata
  40-59  = acne moderata-severa
  0-39   = acne severa con molte lesioni

Regole:
- Usa SEMPRE numeri interi (es. 63, 87).
- La stessa immagine deve produrre SEMPRE gli stessi punteggi.
- Rispondi SOLO con il JSON, senza testo aggiuntivo."""

PROMPT_B = """Sei un assistente dermatologico che valuta SOLO la PIGMENTAZIONE della pelle del viso:
macchie (iperpigmentazione) e occhiaie (scurimento dell'area perioculare).

Devi restituire ESCLUSIVAMENTE un JSON con 2 campi (0-100):
{
  "macchie": ...,
  "occhiaie": ...
}

IMPORTANTE:
- Ignora completamente rughe, pori, acne, texture e luminosità globale.
- Concentrati SOLO su colore e uniformità della pigmentazione.

Scala (0-100, 100 = situazione migliore, cioè meno problema visibile):

- Macchie (iperpigmentazione generale: macchie solari, discromie):
  90-100 = tono molto uniforme, nessuna macchia visibile
  75-89  = qualche piccola macchia isolata
  60-74  = iperpigmentazione lieve-moderata in alcune aree
  40-59  = macchie evidenti e diffuse
  0-39   = iperpigmentazione marcata e molto diffusa

- Occhiaie (area sotto gli occhi, colore e ombra):
  90-100 = nessuna occhiaia visibile, area perioculare chiara
  75-89  = leggero scurimento, occhiaie appena accennate
  60-74  = occhiaie chiaramente visibili, ma non molto profonde
  40-59  = occhiaie marcate, colore scuro o bluastro evidente
  0-39   = occhiaie molto profonde e scure

Ignora ombre da illuminazione:
- Non confondere un'ombra di luce con un'occhiaia: se il bordo dell'ombra coincide con la direzione della luce, penalizza meno il punteggio.

Regole:
- Valuta macchie e occhiaie in modo INDIPENDENTE.
- Usa solo interi.
- Rispondi SOLO con il JSON, senza testo aggiuntivo."""

PROMPT_C = """Sei un assistente dermatologico che valuta l'IDRATAZIONE superficiale e la PULIZIA globale
della pelle del viso in una foto.

Devi restituire ESCLUSIVAMENTE questo JSON:
{
  "disidratazione": ...,
  "pelle_pulita_percent": ...
}

Definizioni:
- Disidratazione = quanto la pelle appare secca, che tira, con micro-linee da secchezza,
  opaca e priva di rimbalzo. Valore ALTO = pelle ben idratata (poca disidratazione).
- Pelle pulita % = percezione globale di pelle libera da impurità visibili (sebo in eccesso,
  sporco, make-up pesante non rimosso).

Scala (0-100, 100 = situazione ideale):

- Disidratazione:
  90-100 = pelle visibilmente rimpolpata, liscia, con luminosità sana, nessuna micro-squama
  75-89  = pelle generalmente idratata, lievi segni di secchezza in alcune aree
  60-74  = disidratazione lieve-moderata (opacità, micro-linee da secchezza)
  40-59  = pelle secca, evidenti aree ruvide/spente
  0-39   = pelle molto disidratata, aspetto spento e ruvido

- Pelle pulita percentuale:
  90-100 = pelle molto pulita, senza film lucido eccessivo né residui visibili
  75-89  = pelle pulita con leggero sebo naturale
  60-74  = qualche zona lucida o leggermente congestionata
  40-59  = numerose zone lucide/untuose o residui evidenti
  0-39   = pelle poco pulita, sebo/impurità marcate

Regole:
- NON giudicare rughe, pori, macchie, occhiaie o acne: questi parametri sono già valutati altrove.
- Concentrati sull'aspetto globale (luminosità diffusa, uniformità, sebo vs secchezza).
- Usa solo numeri interi.
- Rispondi SOLO con il JSON, senza testo aggiuntivo."""

def call_groq(prompt, img_b64, seed, retries=2):
    last_error = None
    for attempt in range(retries + 1):
        try:
            completion = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                        ],
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0,
                seed=seed,
            )
            return json.loads(completion.choices[0].message.content)
        except Exception as e:
            last_error = e
            if attempt < retries:
                import time
                time.sleep(1)
    raise last_error

@app.get("/health")
async def health():
    return {"status": "online", "mode": "Maverick-Groq-OpenCV", "api_key": bool(os.environ.get("GROQ_API_KEY"))}

@app.post("/analyze")
@app.post("/analyze-dermoscope")
async def analyze(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        img_b64, processed_bytes = prepare_image_opencv(img_bytes)
        seed = image_seed(processed_bytes)

        result_a = call_groq(PROMPT_A, img_b64, seed)
        result_b = call_groq(PROMPT_B, img_b64, seed + 1)
        result_c = call_groq(PROMPT_C, img_b64, seed + 2)

        beauty_scores = {}
        beauty_scores.update(result_a)
        beauty_scores.update(result_b)
        beauty_scores.update(result_c)

        required_fields = ["rughe", "pori", "macchie", "occhiaie", "disidratazione", "acne", "pelle_pulita_percent"]
        for field in required_fields:
            if field not in beauty_scores:
                beauty_scores[field] = 50
            beauty_scores[field] = max(0, min(100, int(beauty_scores[field])))

        ragionamento = (
            f"Texture: rughe={beauty_scores['rughe']}, pori={beauty_scores['pori']}, acne={beauty_scores['acne']}. "
            f"Colore: macchie={beauty_scores['macchie']}, occhiaie={beauty_scores['occhiaie']}. "
            f"Idratazione: {beauty_scores['disidratazione']}. Salute generale: {beauty_scores['pelle_pulita_percent']}."
        )

        return {"beauty_scores": beauty_scores, "ragionamento": ragionamento}

    except Exception as e:
        return {"error": str(e), "status": "failed"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
