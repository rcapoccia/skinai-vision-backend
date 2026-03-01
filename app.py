import os
import base64
import json
import io
import hashlib
import numpy as np
import cv2
from PIL import Image, ImageFilter
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from typing import Optional

app = FastAPI(title="SkinGlow AI - Hybrid CV+Groq v3 (foto domina)")

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
    Scala: 100 = pori quasi invisibili, 0 = pori molto dilatati.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return 70

    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(float)

    nose_top = int(h * 0.35)
    nose_bot = int(h * 0.70)
    nose_left = int(w * 0.25)
    nose_right = int(w * 0.75)
    nose_roi = gray[nose_top:nose_bot, nose_left:nose_right]
    if nose_roi.size == 0:
        nose_roi = gray

    blur_small = cv2.GaussianBlur(nose_roi, (3, 3), 0.8)
    blur_large = cv2.GaussianBlur(nose_roi, (15, 15), 4)
    dog = blur_small - blur_large
    dog_var = dog.var()

    pori_score = int(np.interp(dog_var, [0, 50, 200, 800, 2000], [95, 85, 70, 50, 30]))
    return int(np.clip(pori_score, 10, 100))


# ─────────────────────────────────────────────
# GROQ — PUNTEGGI VISIVI PURI
# ─────────────────────────────────────────────

PROMPT_PUNTEGGI = """Sei un dermatologo AI esperto. Analizza il viso nella foto.
Valuta ESATTAMENTE questi 6 parametri dermatologici basandoti SOLO su ciò che vedi nella foto.
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

DISIDRATAZIONE (secchezza, opacità, mancanza di luminosità):
  90-100 = pelle luminosa, idratata, elastica
  75-89  = pelle normalmente idratata
  60-74  = lieve disidratazione, qualche zona opaca
  40-59  = disidratazione moderata, pelle opaca
  0-39   = pelle molto secca, opaca, micro-rughe da secchezza

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


# ─────────────────────────────────────────────
# GROQ — ROUTINE + ALERT (usa punteggi + quiz)
# ─────────────────────────────────────────────

def build_prompt_routine(punteggi: dict, quiz: dict) -> str:
    """Costruisce il prompt per Groq che genera routine personalizzata e alert."""

    punteggi_str = json.dumps(punteggi, ensure_ascii=False)
    quiz_str = json.dumps(quiz, ensure_ascii=False) if quiz else "{}"

    return f"""Sei un dermatologo e cosmetologo esperto. Hai a disposizione:

1. PUNTEGGI VISIVI REALI (rilevati dalla foto, immutabili):
{punteggi_str}
(scala 0-100, dove 100 = ottimo, 0 = problema grave)

2. PROFILO PERSONALE DICHIARATO DAL CLIENTE:
{quiz_str}

Il tuo compito è:
A) Identificare eventuali CONFLITTI tra foto e dichiarato (es. foto mostra pelle secca ma cliente dice "pelle grassa")
B) Generare una ROUTINE SKINCARE personalizzata basata sui punteggi reali, usando il profilo per calibrare i prodotti
C) Aggiungere CONSIGLI DI PREVENZIONE basati sul profilo (es. se dorme poco, prevenire le occhiaie anche se non visibili ora)

REGOLA FONDAMENTALE: i punteggi visivi NON cambiano mai. Il profilo serve solo per personalizzare i consigli.

Restituisci ESCLUSIVAMENTE questo JSON valido, senza testo aggiuntivo:
{{
  "routine_mattina": ["step1", "step2", "step3", "step4"],
  "routine_sera": ["step1", "step2", "step3", "step4", "step5"],
  "consigli_lifestyle": ["consiglio1", "consiglio2", "consiglio3"],
  "alerts": ["alert1"],
  "quiz_contesto": "Una frase che spiega come il profilo ha personalizzato la routine"
}}

LINEE GUIDA ROUTINE:
- Se rughe < 70: includi retinolo sera (0.25-0.5% se < 60, 0.5-1% se < 40)
- Se disidratazione < 70: enfatizza idratanti, acido ialuronico
- Se occhiaie < 70: contorno occhi con caffeina + vitamina K
- Se acne < 80: niacinamide, salicilico, no prodotti occlusivi
- Se macchie < 75: vitamina C mattina, SPF 50+ obbligatorio
- Tipo pelle secca: prodotti cremosi, no alcol, no salicilico aggressivo
- Tipo pelle grassa: gel leggeri, niacinamide, SPF non comedogenico
- Poco sonno (< 6h): aggiungi contorno occhi preventivo anche se occhiaie ok
- Stress alto: aggiungi adattogeni, routine anti-stress

ALERTS (solo se c'è un vero conflitto o anomalia):
- Conflitto foto vs dichiarato (es. "La foto mostra pelle tendenzialmente secca, ma hai dichiarato pelle grassa: fidati della foto")
- Fattore di rischio futuro (es. "Con 4h di sonno, le occhiaie potrebbero peggiorare: prevenzione raccomandata")
- Anomalia clinica (es. "Rilevata iperpigmentazione significativa: consulta un dermatologo")

Se non ci sono conflitti o anomalie, restituisci "alerts": [].
"""


def call_groq_punteggi(img_b64, seed, retries=2):
    last_error = None
    for attempt in range(retries + 1):
        try:
            completion = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": PROMPT_PUNTEGGI},
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


def call_groq_routine(punteggi: dict, quiz: dict, retries=2):
    prompt = build_prompt_routine(punteggi, quiz)
    last_error = None
    for attempt in range(retries + 1):
        try:
            completion = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
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
        "mode": "Hybrid-CV(pori)+Groq(6params)+Groq(routine) v3",
        "api_key": bool(os.environ.get("GROQ_API_KEY"))
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """Endpoint legacy: solo punteggi visivi puri, nessun quiz."""
    try:
        img_bytes = await file.read()
        img_b64, processed_bytes = prepare_image(img_bytes)
        seed = image_seed(processed_bytes)
        pori_score = cv_analyze_pori(processed_bytes)
        groq_scores = call_groq_punteggi(img_b64, seed)

        beauty_scores = {}
        beauty_scores.update(groq_scores)
        beauty_scores["pori"] = pori_score

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
            "ragionamento": ragionamento,
            "status": "success",
            "result": beauty_scores
        }

    except Exception as e:
        return {"error": str(e), "status": "failed"}


@app.post("/analyze_full")
async def analyze_full(
    file: UploadFile = File(...),
    quiz: Optional[str] = Form(None)
):
    """
    Endpoint principale v3.
    Riceve foto + quiz (JSON string opzionale).
    Restituisce punteggi visivi PURI (immutati) + routine personalizzata + alert.
    Il questionario NON modifica mai i punteggi.
    """
    try:
        img_bytes = await file.read()
        img_b64, processed_bytes = prepare_image(img_bytes)
        seed = image_seed(processed_bytes)

        # Parse quiz
        quiz_data = {}
        if quiz:
            try:
                quiz_data = json.loads(quiz)
            except Exception:
                quiz_data = {}

        # 1. Punteggi visivi puri
        pori_score = cv_analyze_pori(processed_bytes)
        groq_scores = call_groq_punteggi(img_b64, seed)

        punteggi = {}
        punteggi.update(groq_scores)
        punteggi["pori"] = pori_score

        required_fields = ["rughe", "pori", "macchie", "occhiaie", "disidratazione", "acne", "pelle_pulita_percent"]
        for field in required_fields:
            if field not in punteggi:
                punteggi[field] = 50
            punteggi[field] = max(0, min(100, int(punteggi[field])))

        # 2. Routine + alert (usa punteggi reali + quiz, non modifica i punteggi)
        routine_data = call_groq_routine(punteggi, quiz_data)

        # 3. Salute complessiva (media ponderata dei punteggi visivi puri)
        salute_complessiva = round(
            punteggi["rughe"] * 0.15 +
            punteggi["pori"] * 0.15 +
            punteggi["macchie"] * 0.15 +
            punteggi["occhiaie"] * 0.15 +
            punteggi["disidratazione"] * 0.20 +
            punteggi["acne"] * 0.20
        )

        return {
            "punteggi": punteggi,
            "salute_complessiva": salute_complessiva,
            "routine_mattina": routine_data.get("routine_mattina", []),
            "routine_sera": routine_data.get("routine_sera", []),
            "consigli_lifestyle": routine_data.get("consigli_lifestyle", []),
            "alerts": routine_data.get("alerts", []),
            "quiz_contesto": routine_data.get("quiz_contesto", ""),
            "ragionamento": (
                f"CV(pori)={punteggi['pori']}. "
                f"AI: rughe={punteggi['rughe']}, macchie={punteggi['macchie']}, "
                f"occhiaie={punteggi['occhiaie']}, disidratazione={punteggi['disidratazione']}, "
                f"acne={punteggi['acne']}."
            ),
            "status": "success"
        }

    except Exception as e:
        return {"error": str(e), "status": "failed"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
