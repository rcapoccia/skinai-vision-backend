#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import time
import uuid
import zipfile
import json
import asyncio
import requests
from typing import Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# ─────────────────────────────────────────────
# CONFIGURAZIONE APP E STORAGE IN-MEMORY
# ─────────────────────────────────────────────

app = FastAPI(title="SkinAI - Perfect Corp Skin Analysis v6 (Async)")

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

PERFECT_CORP_API_KEY = os.environ.get("PERFECT_CORP_API_KEY")
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
    try:
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
    except Exception as e:
        # Non usare HTTPException qui perché siamo in un background task
        print(f"[ERRORE] Preparazione immagine fallita: {e}")
        raise

# ─────────────────────────────────────────────
# WORKER ASINCRONO PER PERFECT CORP
# ─────────────────────────────────────────────

async def run_perfect_corp_workflow(image_bytes: bytes, internal_task_id: str):
    """Task in background che esegue l'intero workflow Perfect Corp."""
    global tasks
    try:
        # 1. Prepara l'immagine
        tasks[internal_task_id]["perfect_status"] = "preparing_image"
        prepared_image = prepare_image(image_bytes)
        file_size = len(prepared_image)

        # 2. Ottieni pre-signed URL
        tasks[internal_task_id]["perfect_status"] = "getting_presigned_url"
        headers = {
            "Authorization": f"Bearer {PERFECT_CORP_API_KEY}",
            "Content-Type": "application/json"
        }
        payload_file = {
            "files": [{
                "content_type": "image/jpeg",
                "file_name": "photo.jpg",
                "file_size": file_size
            }]
        }
        resp1 = requests.post(f"{BASE_URL}/s2s/v2.0/file/skin-analysis", headers=headers, json=payload_file, timeout=30)
        resp1.raise_for_status()
        data1 = resp1.json()
        file_id = data1["data"]["files"][0]["file_id"]
        upload_url = data1["data"]["files"][0]["requests"][0]["url"]
        upload_headers = data1["data"]["files"][0]["requests"][0].get("headers", {})

        # 3. Upload su S3
        tasks[internal_task_id]["perfect_status"] = "uploading_to_s3"
        upload_headers["Content-Type"] = "image/jpeg"
        resp2 = requests.put(upload_url, headers=upload_headers, data=prepared_image, timeout=60)
        resp2.raise_for_status()

        # 4. Avvia il task di analisi
        tasks[internal_task_id]["perfect_status"] = "starting_analysis_task"
        payload_task = {"src_file_id": file_id, "dst_actions": DST_ACTIONS_HD}
        resp3 = requests.post(f"{BASE_URL}/s2s/v2.0/task/skin-analysis", headers=headers, json=payload_task, timeout=30)
        resp3.raise_for_status()
        perfect_task_id = resp3.json()["data"]["task_id"]
        tasks[internal_task_id]["perfect_task_id"] = perfect_task_id

        # 5. Polling del risultato
        tasks[internal_task_id]["perfect_status"] = "polling_result"
        deadline = time.time() + 180  # Timeout di 3 minuti
        while time.time() < deadline:
            await asyncio.sleep(3)
            resp4 = requests.get(f"{BASE_URL}/s2s/v2.0/task/skin-analysis/{perfect_task_id}", headers={"Authorization": f"Bearer {PERFECT_CORP_API_KEY}"}, timeout=30)
            data4 = resp4.json()
            status = data4.get("data", {}).get("status", "unknown")
            tasks[internal_task_id]["perfect_status"] = status
            if status == "success":
                zip_url = data4["data"]["result"]["zip_url"]
                zip_content = requests.get(zip_url, timeout=30).content
                with zipfile.ZipFile(io.BytesIO(zip_content)) as z:
                    with z.open('score_info.json') as f:
                        scores = json.load(f)
                tasks[internal_task_id]["status"] = "success"
                tasks[internal_task_id]["result"] = scores
                return
            elif status in ("failed", "error"):
                raise Exception(f"Perfect Corp task failed: {data4.get('data', {}).get('error')}")
        raise Exception("Polling timeout after 180s")

    except Exception as e:
        print(f"[ERRORE] Task {internal_task_id} fallito: {e}")
        tasks[internal_task_id]["status"] = "failed"
        tasks[internal_task_id]["error"] = str(e)

# ─────────────────────────────────────────────
# ENDPOINTS API
# ─────────────────────────────────────────────

@app.get("/health", summary="Controlla lo stato del servizio")
def health_check():
    return {"status": "ok", "engine": "perfect_corp_v6_async"}

@app.post("/analyze", summary="Avvia l'analisi della pelle e restituisce un task_id")
async def start_analysis(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not PERFECT_CORP_API_KEY:
        raise HTTPException(status_code=500, detail="PERFECT_CORP_API_KEY non configurata")
    
    image_bytes = await file.read()
    internal_task_id = str(uuid.uuid4())

    # Inizializza lo stato del task
    tasks[internal_task_id] = {
        "status": "processing",
        "perfect_status": "queued",
        "created_at": time.time(),
    }

    # Avvia il workflow in background
    background_tasks.add_task(run_perfect_corp_workflow, image_bytes, internal_task_id)

    # Restituisce subito il task_id per il polling
    return {
        "task_id": internal_task_id,
        "status": "processing",
        "poll_url": f"/result/{internal_task_id}"
    }

@app.get("/result/{task_id}", summary="Fa il polling del risultato dell'analisi")
async def get_result(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task non trovato")

    task = tasks[task_id]
    status = task["status"]

    if status == "success":
        return {"status": "success", "result": task["result"]}
    elif status == "failed":
        return {"status": "failed", "error": task.get("error", "Errore sconosciuto")}
    else: # Ancora in elaborazione
        return {
            "status": status,
            "perfect_status": task.get("perfect_status", "pending"),
            "elapsed_time": round(time.time() - task["created_at"], 1),
            "message": "Analisi in corso, riprova tra 5 secondi..."
        }
