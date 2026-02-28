#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import io
import time
import zipfile
import json
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# ─────────────────────────────────────────────
# CONFIGURAZIONE APP
# ─────────────────────────────────────────────

app = FastAPI(title="SkinAI - Perfect Corp Skin Analysis v4")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# VARIABILI E COSTANTI
# ─────────────────────────────────────────────

PERFECT_CORP_API_KEY = os.environ.get("PERFECT_CORP_API_KEY")
BASE_URL = "https://yce-api-01.makeupar.com"
MAX_IMAGE_SIZE = 1920  # Ottimale per Perfect Corp

# Feature HD richieste all'API
DST_ACTIONS_HD = [
    "hd_wrinkle", "hd_pore", "hd_dark_circle", "hd_moisture", "hd_acne",
    "hd_age_spot", "hd_radiance", "hd_oiliness", "hd_texture", "hd_firmness",
    "hd_redness", "hd_eye_bag", "hd_tear_trough"
]

# ─────────────────────────────────────────────
# FUNZIONI HELPER
# ─────────────────────────────────────────────

def prepare_image(image_bytes: bytes) -> bytes:
    """Ridimensiona l'immagine e la converte in JPEG, rispettando i requisiti di Perfect Corp."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = img.size

        # Ridimensiona se il lato più lungo supera MAX_IMAGE_SIZE
        if max(w, h) > MAX_IMAGE_SIZE:
            ratio = MAX_IMAGE_SIZE / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

        # Assicura che il lato corto sia >= 480px (requisito per SD, buona norma per HD)
        min_side = min(img.size)
        if min_side < 480:
            ratio = 480 / min_side
            img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=90)
        return buffer.getvalue()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Errore preparazione immagine: {e}")

# ─────────────────────────────────────────────
# PERFECT CORP API - WORKFLOW
# ─────────────────────────────────────────────

def upload_file_and_get_id(image_bytes: bytes) -> str:
    """Step 1: Carica l'immagine via multipart/form-data e ottiene il file_id."""
    headers = {"Authorization": f"BearerAuthenticationV2 {PERFECT_CORP_API_KEY}"}
    files = {"file": ("photo.jpg", image_bytes, "image/jpeg")}
    url = f"{BASE_URL}/s2s/v2.0/file/skin-analysis"

    try:
        resp = requests.post(url, headers=headers, files=files, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        file_id = data.get("data", {}).get("files", [{}])[0].get("file_id")
        if not file_id:
            raise ValueError("file_id non trovato nella risposta API")
        return file_id
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Errore API (Upload): {e}")
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=500, detail=f"Errore parsing risposta API (Upload): {e}")


def run_skin_analysis_task(file_id: str) -> str:
    """Step 2: Avvia il task di analisi e ottiene il task_id."""
    headers = {
        "Authorization": f"BearerAuthenticationV2 {PERFECT_CORP_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "src_file_id": file_id,
        "dst_actions": DST_ACTIONS_HD,
        "miniserver_args": {"enable_mask_overlay": False}
    }
    url = f"{BASE_URL}/s2s/v2.0/task/skin-analysis"

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        task_id = data.get("data", {}).get("task_id")
        if not task_id:
            raise ValueError("task_id non trovato nella risposta API")
        return task_id
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Errore API (Run Task): {e}")
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=500, detail=f"Errore parsing risposta API (Run Task): {e}")

def poll_for_result(task_id: str, max_wait: int = 60) -> dict:
    """Step 3: Polling del risultato, download dello ZIP e estrazione di score_info.json."""
    headers = {"Authorization": f"BearerAuthenticationV2 {PERFECT_CORP_API_KEY}"}
    url = f"{BASE_URL}/s2s/v2.0/task/skin-analysis/{task_id}"
    deadline = time.time() + max_wait

    while time.time() < deadline:
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json().get("data", {})
            status = data.get("status")

            if status == "done":
                zip_url = data.get("result_url")
                if not zip_url:
                    raise ValueError("result_url non trovato nella risposta API")
                
                zip_resp = requests.get(zip_url, timeout=30)
                zip_resp.raise_for_status()

                with zipfile.ZipFile(io.BytesIO(zip_resp.content)) as z:
                    score_file_name = next((f for f in z.namelist() if f.endswith("score_info.json")), None)
                    if not score_file_name:
                        raise ValueError(f"score_info.json non trovato nello ZIP. File presenti: {z.namelist()}")
                    with z.open(score_file_name) as f:
                        return json.load(f)

            elif status == "failed":
                raise HTTPException(status_code=500, detail=f"Task di analisi fallito: {data.get('error')}")

            time.sleep(2) # Intervallo di polling

        except requests.RequestException as e:
            raise HTTPException(status_code=502, detail=f"Errore API (Polling): {e}")
    raise HTTPException(status_code=408, detail=f"Timeout: Task {task_id} non completato entro {max_wait}s")

# ─────────────────────────────────────────────
# MAPPING PUNTEGGI
# ─────────────────────────────────────────────

def map_scores_to_frontend(score_info: dict) -> dict:
    """Mappa i punteggi HD di Perfect Corp ai nomi usati dal frontend."""
    
    def get_score(key: str, default: int = 50) -> int:
        val = score_info.get(key, {})
        # L'API può restituire direttamente il punteggio o un dizionario
        if isinstance(val, dict):
            raw = val.get("ui_score", val.get("raw_score", default))
        elif isinstance(val, (int, float)):
            raw = val
        else:
            raw = default
        return max(0, min(100, int(raw)))

    return {
        "rughe": get_score("hd_wrinkle"),
        "pori": get_score("hd_pore"),
        "macchie": get_score("hd_age_spot"),
        "occhiaie": get_score("hd_dark_circle"),
        "disidratazione": get_score("hd_moisture"),
        "acne": get_score("hd_acne"),
        "pelle_pulita_percent": get_score("all", 80),
        "radiance": get_score("hd_radiance"),
        "firmness": get_score("hd_firmness"),
        "oiliness": get_score("hd_oiliness"),
        "texture": get_score("hd_texture"),
        "redness": get_score("hd_redness"),
        "skin_age": score_info.get("skin_age", {}).get("ui_score") if isinstance(score_info.get("skin_age"), dict) else score_info.get("skin_age"),
    }


# ─────────────────────────────────────────────
# ENDPOINTS API
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "engine": "perfect_corp_v4"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not PERFECT_CORP_API_KEY:
        raise HTTPException(status_code=500, detail="PERFECT_CORP_API_KEY non configurata sul server")

    # 1. Leggi e prepara l'immagine
    raw_bytes = await file.read()
    image_bytes = prepare_image(raw_bytes)

    # 2. Esegui il workflow Perfect Corp
    file_id = upload_file_and_get_id(image_bytes)
    task_id = run_skin_analysis_task(file_id)
    score_info = poll_for_result(task_id)

    # 3. Mappa i punteggi per il frontend
    beauty_scores = map_scores_to_frontend(score_info)

    # 4. Crea un testo di "ragionamento" per il debug o la visualizzazione
    ragionamento = (
        f"Analisi Perfect Corp (HD): rughe={beauty_scores['rughe']}, "
        f"pori={beauty_scores['pori']}, macchie={beauty_scores['macchie']}, "
        f"occhiaie={beauty_scores['occhiaie']}, idratazione={beauty_scores['disidratazione']}, "
        f"acne={beauty_scores['acne']}."
    )

    return {
        "beauty_scores": beauty_scores,
        "ragionamento": ragionamento,
        "raw_scores": score_info
    }

# ─────────────────────────────────────────────
# ESECUZIONE (per Railway)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
