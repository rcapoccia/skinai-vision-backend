import os
import io
import time
import zipfile
import json
import requests
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageFilter

app = FastAPI(title="SkinAI - Perfect Corp Skin Analysis v3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PERFECT_CORP_API_KEY = os.environ.get("PERFECT_CORP_API_KEY")
BASE_URL = "https://yce-api-01.makeupar.com"
MAX_IMAGE_SIZE = 1920  # Perfect Corp supporta fino a 4096px, ma 1920 è ottimale


# ─────────────────────────────────────────────
# PREPARAZIONE IMMAGINE
# ─────────────────────────────────────────────

def prepare_image(image_bytes: bytes) -> bytes:
    """Ridimensiona l'immagine a max 1920px e la converte in JPEG."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_IMAGE_SIZE:
        ratio = MAX_IMAGE_SIZE / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    # Assicura che il lato corto sia >= 480px (requisito Perfect Corp SD)
    min_side = min(img.size)
    if min_side < 480:
        ratio = 480 / min_side
        img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


# ─────────────────────────────────────────────
# PERFECT CORP API - 3 STEP WORKFLOW
# ─────────────────────────────────────────────

def upload_file(image_bytes: bytes) -> str:
    """Step 1: Carica l'immagine e ottieni il file_id."""
    headers = {"X-API-KEY": PERFECT_CORP_API_KEY}
    files = {"file": ("photo.jpg", image_bytes, "image/jpeg")}
    resp = requests.post(f"{BASE_URL}/v1/file", headers=headers, files=files, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["data"]["file_id"]


def run_skin_analysis(file_id: str) -> str:
    """Step 2: Avvia il task di analisi della pelle e ottieni il task_id."""
    headers = {
        "X-API-KEY": PERFECT_CORP_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "file_id": file_id,
        "enable_mask_overlay": False
    }
    resp = requests.post(
        f"{BASE_URL}/v1/skin-analysis",
        headers=headers,
        json=payload,
        timeout=30
    )
    resp.raise_for_status()
    data = resp.json()
    return data["data"]["task_id"]


def poll_result(task_id: str, max_wait: int = 60) -> dict:
    """Step 3: Polling finché il task è completato, poi scarica e legge lo ZIP."""
    headers = {"X-API-KEY": PERFECT_CORP_API_KEY}
    deadline = time.time() + max_wait

    while time.time() < deadline:
        resp = requests.get(
            f"{BASE_URL}/v1/skin-analysis/{task_id}",
            headers=headers,
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        status = data["data"].get("status")

        if status == "done":
            zip_url = data["data"]["result_url"]
            zip_resp = requests.get(zip_url, timeout=30)
            zip_resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(zip_resp.content)) as z:
                # Il file score_info.json è dentro la cartella skinanalysisResult/
                score_file = next(
                    (f for f in z.namelist() if f.endswith("score_info.json")),
                    None
                )
                if not score_file:
                    raise ValueError("score_info.json non trovato nello ZIP")
                with z.open(score_file) as f:
                    return json.load(f)

        elif status == "failed":
            raise ValueError(f"Task fallito: {data}")

        time.sleep(2)

    raise TimeoutError(f"Task {task_id} non completato entro {max_wait}s")


# ─────────────────────────────────────────────
# MAPPING Perfect Corp → SkinAI scores
# ─────────────────────────────────────────────

def map_scores(score_info: dict) -> dict:
    """
    Mappa i punteggi Perfect Corp (1-100) ai parametri SkinAI.
    Perfect Corp: 100 = pelle perfetta, 1 = pelle con problemi.
    SkinAI usa la stessa scala: 100 = ottimo.

    Per la disidratazione: Perfect Corp 'moisture' = 100 → molto idratata.
    SkinAI 'disidratazione' = 100 → molto idratata (stessa direzione).
    """
    def get(key: str, default: int = 50) -> int:
        val = score_info.get(key, {})
        if isinstance(val, dict):
            raw = val.get("raw_score", val.get("ui_score", default))
        elif isinstance(val, (int, float)):
            raw = val
        else:
            raw = default
        return max(0, min(100, int(raw)))

    return {
        "rughe": get("wrinkle"),
        "pori": get("pore"),
        "macchie": get("age_spot"),
        "occhiaie": get("dark_circle_v2"),
        "disidratazione": get("moisture"),
        "acne": get("acne"),
        "pelle_pulita_percent": get("all"),
        # Parametri bonus Perfect Corp
        "radiance": get("radiance"),
        "firmness": get("firmness"),
        "oiliness": get("oiliness"),
        "texture": get("texture"),
        "redness": get("redness"),
        "skin_age": score_info.get("skin_age", None),
    }


# ─────────────────────────────────────────────
# ENDPOINT
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "engine": "perfect_corp_v3"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        if not PERFECT_CORP_API_KEY:
            return {"error": "PERFECT_CORP_API_KEY non configurata", "status": "failed"}

        # 1. Leggi e prepara l'immagine
        raw_bytes = await file.read()
        image_bytes = prepare_image(raw_bytes)

        # 2. Upload
        file_id = upload_file(image_bytes)

        # 3. Avvia analisi
        task_id = run_skin_analysis(file_id)

        # 4. Polling risultato
        score_info = poll_result(task_id)

        # 5. Mapping punteggi
        beauty_scores = map_scores(score_info)

        ragionamento = (
            f"Perfect Corp AI: rughe={beauty_scores['rughe']}, "
            f"pori={beauty_scores['pori']}, macchie={beauty_scores['macchie']}, "
            f"occhiaie={beauty_scores['occhiaie']}, idratazione={beauty_scores['disidratazione']}, "
            f"acne={beauty_scores['acne']}, salute={beauty_scores['pelle_pulita_percent']}."
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
