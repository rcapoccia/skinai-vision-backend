#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import time
import uuid
import zipfile
import json
from typing import Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# ─────────────────────────────────────────────
# CONFIGURAZIONE APP E STORAGE IN-MEMORY
# ─────────────────────────────────────────────

app = FastAPI(title="SkinAI - Perfect Corp Skin Analysis v6")

# Storage in-memory per i task (adatto per Railway free/hobby tier con 1 worker)
tasks: Dict[str, Dict[str, Any]] = {}

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

PERFECT_CORP_API_KEY = os.environ.get("PERFECT_CORP_API_KEY", "")
BASE_URL = "https://yce-api-01.makeupar.com"
MAX_IMAGE_SIZE = 1920

DST_ACTIONS_HD = [
    "hd_wrinkle", "hd_pore", "hd_dark_circle", "hd_moisture", "hd_acne",
    "hd_age_spot", "hd_radiance", "hd_oiliness", "hd_texture", "hd_firmness",
    "hd_redness", "hd_eye_bag", "hd_tear_trough"
]

# ─────────────────────────────────────────────
# FUNZIONI HELPER
# ─────────────────────────────────────────────

def prepare_image(image_bytes: bytes) -> bytes:
    """Ridimensiona e converte l'immagine in JPEG."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_IMAGE_SIZE:
        ratio = MAX_IMAGE_SIZE / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    min_side = min(img.size)
    if min_side < 480:
        ratio = 480 / min_side
        img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def do_perfect_corp_workflow(image_bytes: bytes, internal_task_id: str):
    """
    Esegue l'intero workflow Perfect Corp in modo sincrono.
    Viene chiamata da BackgroundTasks che la esegue in un threadpool interno
    DOPO aver inviato la risposta HTTP → nessun timeout Railway.
    """
    import requests

    try:
        # 1. Prepara immagine
        tasks[internal_task_id]["perfect_status"] = "preparing_image"
        prepared = prepare_image(image_bytes)
        file_size = len(prepared)

        headers_json = {
            "Authorization": f"Bearer {PERFECT_CORP_API_KEY}",
            "Content-Type": "application/json"
        }

        # 2. Ottieni pre-signed URL da Perfect Corp
        tasks[internal_task_id]["perfect_status"] = "getting_presigned_url"
        payload_file = {
            "files": [{
                "content_type": "image/jpeg",
                "file_name": "photo.jpg",
                "file_size": file_size
            }]
        }
        r1 = requests.post(
            f"{BASE_URL}/s2s/v2.0/file/skin-analysis",
            headers=headers_json,
            json=payload_file,
            timeout=30
        )
        r1.raise_for_status()
        d1 = r1.json()
        file_id = d1["data"]["files"][0]["file_id"]
        upload_url = d1["data"]["files"][0]["requests"][0]["url"]
        upload_headers = dict(d1["data"]["files"][0]["requests"][0].get("headers", {}))

        # 3. Upload binario su S3
        tasks[internal_task_id]["perfect_status"] = "uploading_to_s3"
        upload_headers["Content-Type"] = "image/jpeg"
        r2 = requests.put(upload_url, headers=upload_headers, data=prepared, timeout=60)
        r2.raise_for_status()

        # 4. Avvia task di analisi
        tasks[internal_task_id]["perfect_status"] = "starting_analysis_task"
        payload_task = {
            "src_file_id": file_id,
            "dst_actions": DST_ACTIONS_HD
        }
        r3 = requests.post(
            f"{BASE_URL}/s2s/v2.0/task/skin-analysis",
            headers=headers_json,
            json=payload_task,
            timeout=30
        )
        r3.raise_for_status()
        perfect_task_id = r3.json()["data"]["task_id"]
        tasks[internal_task_id]["perfect_task_id"] = perfect_task_id

        # 5. Polling del risultato (max 3 minuti)
        tasks[internal_task_id]["perfect_status"] = "polling_result"
        headers_auth = {"Authorization": f"Bearer {PERFECT_CORP_API_KEY}"}
        deadline = time.time() + 180
        while time.time() < deadline:
            time.sleep(3)
            r4 = requests.get(
                f"{BASE_URL}/s2s/v2.0/task/skin-analysis/{perfect_task_id}",
                headers=headers_auth,
                timeout=30
            )
            d4 = r4.json()
            status = d4.get("data", {}).get("status", "unknown")
            tasks[internal_task_id]["perfect_status"] = status

            if status == "success":
                zip_url = d4["data"]["result"]["zip_url"]
                zip_bytes = requests.get(zip_url, timeout=30).content
                with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
                    with z.open("score_info.json") as f:
                        scores = json.load(f)
                tasks[internal_task_id]["status"] = "success"
                tasks[internal_task_id]["result"] = scores
                return

            elif status in ("failed", "error"):
                raise Exception(f"Perfect Corp task failed: {d4.get('data', {}).get('error')}")

        raise Exception("Polling timeout after 180s")

    except Exception as e:
        print(f"[ERRORE] Task {internal_task_id}: {e}")
        if internal_task_id in tasks:
            tasks[internal_task_id]["status"] = "failed"
            tasks[internal_task_id]["error"] = str(e)


# ─────────────────────────────────────────────
# ENDPOINTS API
# ─────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "ok", "engine": "perfect_corp_v6"}


@app.post("/analyze")
async def start_analysis(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not PERFECT_CORP_API_KEY:
        raise HTTPException(status_code=500, detail="PERFECT_CORP_API_KEY non configurata")

    image_bytes = await file.read()
    internal_task_id = str(uuid.uuid4())

    tasks[internal_task_id] = {
        "status": "processing",
        "perfect_status": "queued",
        "created_at": time.time(),
    }

    # FastAPI esegue automaticamente le funzioni sincrone in un threadpool interno
    # DOPO aver inviato la risposta HTTP → fire-and-forget, nessun timeout Railway
    background_tasks.add_task(do_perfect_corp_workflow, image_bytes, internal_task_id)

    return {
        "task_id": internal_task_id,
        "status": "processing",
        "poll_url": f"/result/{internal_task_id}"
    }


@app.get("/result/{task_id}")
async def get_result(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task non trovato")

    task = tasks[task_id]
    status = task["status"]

    if status == "success":
        return {"status": "success", "result": task["result"]}
    elif status == "failed":
        return {"status": "failed", "error": task.get("error", "Errore sconosciuto")}
    else:
        # Auto-cleanup task scaduti (>10 minuti)
        if time.time() - task["created_at"] > 600:
            del tasks[task_id]
            raise HTTPException(status_code=404, detail="Task scaduto")
        return {
            "status": status,
            "perfect_status": task.get("perfect_status", "pending"),
            "elapsed_time": round(time.time() - task["created_at"], 1),
            "message": "Analisi in corso, riprova tra 5 secondi..."
        }


@app.delete("/result/{task_id}")
async def cleanup_task(task_id: str):
    """Elimina un task dalla memoria dopo che il frontend ha ricevuto il risultato."""
    if task_id in tasks:
        del tasks[task_id]
    return {"deleted": True}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
