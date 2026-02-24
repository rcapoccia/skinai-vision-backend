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

app = FastAPI(title="SkinGlow AI - Hybrid CV+Groq v2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

MAX_IMAGE_SIZE = 1024


# ─────────────────────────────────────────────
# PREPARAZIONE IMMAGINE
# ─────────────────────────────────────────────

def prepare_image(image_bytes):
    """Resize a max 1024px, sharpen, restituisce base64 + bytes processati."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_IMAGE_SIZE:
        ratio = MAX_IMAGE_SIZE / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    img = img.filter(ImageFilter.SHARPEN)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    processed_bytes = buffer.getvalue()
    return base64.b64encode(processed_bytes).decode('utf-8'), processed_bytes


def image_seed(image_bytes):
    return int(hashlib.md5(image_bytes).hexdigest()[:8], 16)


# ─────────────────────────────────────────────
# CV CLASSICO — SOLO PORI
# ─────────────────────────────────────────────

def cv_analyze_pori(image_bytes):
    """
    Analisi CV classica per i pori usando Difference of Gaussians (DoG).
    I pori dilatati creano micro-texture ad alta frequenza nella zona naso/guance.
    Scala: 100 = pori quasi invisibili, 0 = pori molto dilatati.
    
    Calibrazione empirica su 3 foto di riferimento:
    - dog_var ~64  → pori poco visibili → score ~83
    - dog_var ~80  → pori poco visibili → score ~81
    - dog_var ~162 → pori moderati      → score ~73
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return 70

    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(float)

    # Zona naso + guance centrali (35-70% altezza, 25-75% larghezza)
    nose_top = int(h * 0.35)
    nose_bot = int(h * 0.70)
    nose_left = int(w * 0.25)
    nose_right = int(w * 0.75)
    nose_roi = gray[nose_top:nose_bot, nose_left:nose_right]
    if nose_roi.size == 0:
        nose_roi = gray

    # Difference of Gaussians: cattura frequenze medie (pori, non rughe profonde)
    blur_small = cv2.GaussianBlur(nose_roi, (3, 3), 0.8)
    blur_large = cv2.GaussianBlur(nose_roi, (15, 15), 4)
    dog = blur_small - blur_large
    dog_var = dog.var()

    # Calibrazione: dog_var bassa = pelle liscia/pori chiusi, alta = pori dilatati
    pori_score = int(np.interp(dog_var, [0, 50, 200, 800, 2000], [95, 85, 70, 50, 30]))
    return int(np.clip(pori_score, 10, 100))


# ─────────────────────────────────────────────
# GROQ — TUTTI I PARAMETRI TRANNE PORI
# ─────────────────────────────────────────────

PROMPT_GROQ = """Sei un dermatologo AI esperto. Analizza il viso nella foto.
Valuta ESATTAMENTE questi 6 parametri dermatologici.
NON valutare i pori (già analizzati separatamente con metodo strumentale).

Restituisci ESCLUSIVAMENTE questo JSON, senza testo aggiuntivo:
{
    "rughe": <0-100>,
    "macchie": <0-100>,
    "occhiaie": <0-100>,
    "disidratazione": <0-100>,
    "acne": <0-100>,
    "pelle_pulita_percent": <0-100>
}

SCALA (100 = perfetto/nessun problema, 0 = problema molto grave):

RUGHE (linee di espressione, rughe profonde, solchi):
  90-100 = nessuna ruga visibile, pelle liscia
  75-89  = leggere linee di espressione (angoli occhi, fronte)
  60-74  = rughe visibili su fronte, contorno occhi, naso-labbiale
  40-59  = rughe marcate su più aree del viso
  0-39   = rughe profonde e diffuse (solchi profondi, pelle molto segnata)

MACCHIE (iperpigmentazione, discromie, macchie solari, melasma):
  90-100 = tono uniforme, nessuna macchia visibile
  75-89  = qualche piccola macchia isolata
  60-74  = iperpigmentazione lieve-moderata
  40-59  = macchie evidenti e diffuse
  0-39   = iperpigmentazione marcata e molto diffusa

OCCHIAIE (colorazione scura/bluastra/violacea sotto gli occhi):
  90-100 = nessuna occhiaia, zona sotto gli occhi luminosa
  75-89  = leggero scurimento sotto gli occhi
  60-74  = occhiaie lievi ma visibili
  40-59  = occhiaie moderate, colorazione evidente
  0-39   = occhiaie marcate, colorazione scura/violacea intensa
  IMPORTANTE: valuta SOLO la colorazione sotto gli occhi, non le rughe intorno.
  La pelle anziana può avere occhiaie moderate (40-59) anche se il tono generale è scuro.

DISIDRATAZIONE (secchezza, opacità, mancanza di luminosità):
  90-100 = pelle luminosa, idratata, elastica
  75-89  = pelle normalmente idratata
  60-74  = lieve disidratazione, qualche zona opaca
  40-59  = disidratazione moderata, pelle opaca
  0-39   = pelle molto secca, opaca, micro-rughe da secchezza
  IMPORTANTE: la pelle anziana con rughe profonde e aspetto opaco deve avere score 20-45.
  La pelle giovane con aspetto luminoso deve avere score 70-90.

ACNE (lesioni attive, brufoli, papule, pustole, infiammazioni):
  90-100 = nessuna lesione attiva, pelle pulita
  75-89  = poche lesioni isolate o comedoni
  60-74  = acne lieve-moderata
  40-59  = acne moderata-severa
  0-39   = acne severa con molte lesioni attive

PELLE PULITA %:
  90-100 = pelle molto pulita, senza sebo eccessivo
  75-89  = pelle pulita con leggero sebo naturale
  60-74  = qualche zona lucida o congestionata
  40-59  = numerose zone lucide o residui evidenti
  0-39   = pelle poco pulita, sebo marcato

Usa SOLO numeri interi. La stessa immagine deve produrre SEMPRE gli stessi punteggi."""


def call_groq(img_b64, seed, retries=2):
    last_error = None
    for attempt in range(retries + 1):
        try:
            completion = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": PROMPT_GROQ},
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


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "online",
        "mode": "Hybrid-CV(pori)+Groq(6params) v2",
        "api_key": bool(os.environ.get("GROQ_API_KEY"))
    }


@app.post("/analyze")
@app.post("/analyze-dermoscope")
async def analyze(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()

        # 1. Prepara immagine (resize + sharpen)
        img_b64, processed_bytes = prepare_image(img_bytes)

        # 2. Seed deterministico sull'immagine processata
        seed = image_seed(processed_bytes)

        # 3. CV classico per SOLI PORI (calibrato e testato)
        pori_score = cv_analyze_pori(processed_bytes)

        # 4. Groq per rughe, macchie, occhiaie, disidratazione, acne, pelle_pulita_percent
        groq_scores = call_groq(img_b64, seed)

        # 5. Merge risultati
        beauty_scores = {}
        beauty_scores.update(groq_scores)
        beauty_scores["pori"] = pori_score  # Override con CV (più accurato)

        # 6. Validazione e clamp 0-100
        required_fields = ["rughe", "pori", "macchie", "occhiaie", "disidratazione", "acne", "pelle_pulita_percent"]
        for field in required_fields:
            if field not in beauty_scores:
                beauty_scores[field] = 50
            beauty_scores[field] = max(0, min(100, int(beauty_scores[field])))

        ragionamento = (
            f"CV(pori)={beauty_scores['pori']}. "
            f"AI: rughe={beauty_scores['rughe']}, macchie={beauty_scores['macchie']}, "
            f"occhiaie={beauty_scores['occhiaie']}, disidratazione={beauty_scores['disidratazione']}, "
            f"acne={beauty_scores['acne']}, pelle_pulita={beauty_scores['pelle_pulita_percent']}."
        )

        return {
            "beauty_scores": beauty_scores,
            "ragionamento": ragionamento
        }

    except Exception as e:
        return {"error": str(e), "status": "failed"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
