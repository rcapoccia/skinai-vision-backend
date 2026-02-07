from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import base64
import io
import re
import json
import os
import numpy as np
import cv2

app = FastAPI(title="SkinGlow AI - Beauty Advisor + Dermoscopio")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_groq_client():
    """Inizializza il client Groq con la chiave API."""
    from groq import Groq
    return Groq(api_key=os.environ.get("GROQ_API_KEY"))


# ============================================================
# DERMOSCOPIO VIRTUALE (OpenCV)
# ============================================================

def dermoscope_effect(image_bytes: bytes) -> tuple:
    """
    Simula un dermoscopio virtuale con tecniche di imaging avanzate:
    1. Cross-polarizzazione simulata (CLAHE su canale L)
    2. Unsharp mask per micro-texture (rughe/pori nascosti)
    3. Hue boost sub-cutaneo (pori, vasi)
    4. Denoise per rimuovere rumore fine
    Restituisce: (derm_bytes, derm_base64)
    """
    # Decode immagine
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Impossibile decodificare l'immagine")

    # 1. CROSS-POLARIZZAZIONE (riduce riflessi superficiali, rivela sub-surface)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    # 2. UNSHARP MASK micro-texture (rivela rughe/pori nascosti)
    gaussian = cv2.GaussianBlur(l_enhanced, (0, 0), 2.0)
    unsharp = cv2.addWeighted(l_enhanced, 1.3, gaussian, -0.3, 0)

    # 3. HUE BOOST sub-skin (pori verdi, vasi blu diventano più visibili)
    a_boosted = np.clip(a.astype(np.float32) * 1.15, 0, 255).astype(np.uint8)
    b_boosted = np.clip(b.astype(np.float32) * 1.08, 0, 255).astype(np.uint8)
    lab_derm = cv2.merge([unsharp, a_boosted, b_boosted])

    # 4. DENOISE peli/fine noise
    img_derm = cv2.cvtColor(lab_derm, cv2.COLOR_LAB2BGR)
    img_derm = cv2.medianBlur(img_derm, 3)

    # Encode per Groq
    _, derm_buffer = cv2.imencode('.jpg', img_derm, [cv2.IMWRITE_JPEG_QUALITY, 90])
    derm_bytes = derm_buffer.tobytes()

    # Base64 per frontend debug
    derm_b64 = base64.b64encode(derm_bytes).decode()

    return derm_bytes, derm_b64


# ============================================================
# ARMOCROMIA ENGINE
# ============================================================

def calcola_armocromia(questionnaire: dict) -> dict:
    """Calcola la stagione armocromia basata sulle risposte del questionario."""
    punti_caldo = 0
    punti_freddo = 0
    intensita_alta = 0
    intensita_bassa = 0

    vene = questionnaire.get("vene_polso", "")
    if "Verdi" in vene:
        punti_caldo += 3
    elif "Blu" in vene:
        punti_freddo += 3
    else:
        punti_caldo += 1
        punti_freddo += 1

    abbronzatura = questionnaire.get("abbronzatura", "")
    if "dorata" in abbronzatura.lower() or "facile" in abbronzatura.lower():
        punti_caldo += 2
    elif "rossore" in abbronzatura.lower() or "difficile" in abbronzatura.lower():
        punti_freddo += 2
    elif "non mi" in abbronzatura.lower():
        punti_freddo += 1

    capelli = questionnaire.get("hair_color", "").lower()
    if any(c in capelli for c in ["rosso", "ramato", "castano scuro"]):
        punti_caldo += 1
        intensita_alta += 1
    elif any(c in capelli for c in ["nero"]):
        intensita_alta += 2
    elif any(c in capelli for c in ["biondo chiaro", "grigio", "bianco"]):
        punti_freddo += 1
        intensita_bassa += 1
    elif "biondo scuro" in capelli or "castano chiaro" in capelli:
        intensita_bassa += 1

    occhi = questionnaire.get("eye_color", "").lower()
    if any(c in occhi for c in ["marrone scuro", "nero"]):
        intensita_alta += 1
        punti_caldo += 1
    elif any(c in occhi for c in ["blu", "celeste", "grigio"]):
        punti_freddo += 1
        intensita_bassa += 1
    elif any(c in occhi for c in ["verde", "nocciola", "marrone chiaro"]):
        punti_caldo += 1

    sottotono = "Caldo" if punti_caldo > punti_freddo else "Freddo" if punti_freddo > punti_caldo else "Neutro"

    if sottotono == "Caldo":
        if intensita_alta >= intensita_bassa:
            stagione = "Autunno"
            sotto_stagione = "Autunno Caldo"
            palette = ["Terracotta", "Arancio bruciato", "Oro antico", "Verde oliva", "Marrone cioccolato", "Senape"]
            makeup = {"fondotinta": "Sottotono dorato/pesca", "blush": "Pesca, terracotta", "labbra": "Nude caldo, mattone, arancio bruciato", "occhi": "Bronzo, rame, verde oliva, marrone caldo"}
        else:
            stagione = "Primavera"
            sotto_stagione = "Primavera Calda"
            palette = ["Pesca", "Corallo", "Oro chiaro", "Verde mela", "Azzurro caldo", "Avorio"]
            makeup = {"fondotinta": "Sottotono pesca/dorato chiaro", "blush": "Pesca chiaro, corallo", "labbra": "Corallo, pesca, rosa caldo", "occhi": "Pesca, oro chiaro, verde chiaro, marrone chiaro"}
    else:
        if intensita_alta >= intensita_bassa:
            stagione = "Inverno"
            sotto_stagione = "Inverno Freddo"
            palette = ["Nero", "Bianco puro", "Fucsia", "Blu royal", "Rosso ciliegia", "Argento"]
            makeup = {"fondotinta": "Sottotono rosa/neutro freddo", "blush": "Rosa freddo, berry", "labbra": "Rosso ciliegia, fucsia, berry, rosa freddo", "occhi": "Grigio, argento, blu navy, prugna"}
        else:
            stagione = "Estate"
            sotto_stagione = "Estate Fredda"
            palette = ["Rosa antico", "Lavanda", "Grigio perla", "Azzurro polvere", "Malva", "Argento chiaro"]
            makeup = {"fondotinta": "Sottotono rosa/beige freddo", "blush": "Rosa antico, malva", "labbra": "Rosa antico, malva, berry chiaro", "occhi": "Grigio, lavanda, rosa antico, taupe"}

    return {"stagione": stagione, "sotto_stagione": sotto_stagione, "sottotono": sottotono, "palette_colori": palette, "consigli_makeup": makeup}


# ============================================================
# LINK PRODOTTI AMAZON DINAMICI
# ============================================================

PRINCIPI_ATTIVI = {
    "Disidratazione": {"attivo": "Acido Ialuronico", "query_base": "acido ialuronico siero"},
    "Rughe": {"attivo": "Retinolo", "query_base": "retinolo crema anti rughe"},
    "Acne": {"attivo": "Niacinamide", "query_base": "niacinamide siero acne"},
    "Pori/Macchie": {"attivo": "Vitamina C", "query_base": "vitamina c siero antimacchie"},
    "Occhiaie": {"attivo": "Caffeina + Vitamina K", "query_base": "contorno occhi caffeina occhiaie"},
    "Sensibilità": {"attivo": "Ceramidi + Centella", "query_base": "crema lenitiva ceramidi pelle sensibile"},
    "Luminosità": {"attivo": "Vitamina C + AHA", "query_base": "vitamina c siero luminosita"},
    "Antietà": {"attivo": "Retinolo + Peptidi", "query_base": "retinolo peptidi anti age"},
    "Idratazione": {"attivo": "Acido Ialuronico + Ceramidi", "query_base": "acido ialuronico ceramidi idratante"},
    "Controllo acne": {"attivo": "Acido Salicilico + Niacinamide", "query_base": "acido salicilico niacinamide acne"},
    "Uniformità tono": {"attivo": "Vitamina C + Niacinamide", "query_base": "vitamina c niacinamide uniformante"}
}

def genera_link_amazon(problema: str, tipo_pelle: str, obiettivo: str = "") -> list:
    """Genera link Amazon dinamici basati su problema, tipo pelle e obiettivo."""
    prodotti = []
    if problema in PRINCIPI_ATTIVI:
        info = PRINCIPI_ATTIVI[problema]
        query = f"{info['query_base']} pelle {tipo_pelle.lower()}"
        prodotti.append({"problema": problema, "principio_attivo": info["attivo"], "query": query, "link_amazon": f"https://www.amazon.it/s?k={query.replace(' ', '+')}"})
    if obiettivo and obiettivo in PRINCIPI_ATTIVI and obiettivo != problema:
        info = PRINCIPI_ATTIVI[obiettivo]
        query = f"{info['query_base']} pelle {tipo_pelle.lower()}"
        prodotti.append({"problema": obiettivo, "principio_attivo": info["attivo"], "query": query, "link_amazon": f"https://www.amazon.it/s?k={query.replace(' ', '+')}"})
    spf_query = f"crema solare viso SPF50 pelle {tipo_pelle.lower()}"
    prodotti.append({"problema": "Protezione solare (essenziale)", "principio_attivo": "SPF 50+", "query": spf_query, "link_amazon": f"https://www.amazon.it/s?k={spf_query.replace(' ', '+')}"})
    return prodotti


# ============================================================
# ROUTINE PERSONALIZZATA
# ============================================================

def genera_routine(scores: dict, questionnaire: dict) -> dict:
    """Genera routine skincare personalizzata. Scala invertita: score basso = problema maggiore."""
    routine_attuale = questionnaire.get("routine_attuale", "Nessuna")
    tipo_pelle = questionnaire.get("skin_type", "Mista")
    problema = questionnaire.get("problema_principale", "")

    mattina = ["Detergente delicato"]
    if scores.get("disidratazione", 100) < 60 or tipo_pelle in ["Secca", "Sensibile"]:
        mattina.append("Siero acido ialuronico")
    if scores.get("macchie", 100) < 70 or problema == "Luminosità":
        mattina.append("Siero vitamina C")
    mattina.append("Crema idratante")
    mattina.append("SPF 50 (sempre!)")

    sera = []
    if questionnaire.get("makeup_frequency", "") in ["Quotidiano leggero", "Quotidiano completo"]:
        sera.append("Doppia detersione (olio + gel)")
    else:
        sera.append("Detergente delicato")
    if scores.get("acne", 100) < 70 or tipo_pelle == "Grassa":
        sera.append("Tonico con niacinamide")
    if scores.get("rughe", 100) < 60 and routine_attuale in ["Completa (sieri+SPF)", "Pro (retinolo/acidi)"]:
        sera.append("Retinolo (2-3 volte/settimana)")
    elif scores.get("rughe", 100) < 60:
        sera.append("Siero peptidi anti-age")
    if scores.get("occhiaie", 100) < 70:
        sera.append("Contorno occhi con caffeina")
    sera.append("Crema notte nutriente")

    settimanale = []
    if scores.get("pori", 100) < 70 or tipo_pelle in ["Grassa", "Mista"]:
        settimanale.append("Maschera argilla (1x/settimana)")
    if scores.get("disidratazione", 100) < 60:
        settimanale.append("Maschera idratante (2x/settimana)")
    if routine_attuale in ["Completa (sieri+SPF)", "Pro (retinolo/acidi)"]:
        settimanale.append("Esfoliante AHA/BHA (1-2x/settimana)")

    return {"mattina": mattina, "sera": sera, "settimanale": settimanale, "nota": f"Routine calibrata per pelle {tipo_pelle.lower()}, livello {routine_attuale.lower()}"}


# ============================================================
# CALIBRAZIONE QUIZ (POST-PROCESSING)
# ============================================================

def calibrate_beauty_scores(scores: dict, quiz: dict) -> dict:
    """Calibra i punteggi beauty usando le risposte del questionario. 100=perfetto, 0=problematico."""
    calibrated = scores.copy()

    age = quiz.get("age", 0)
    try:
        age = int(age)
    except (ValueError, TypeError):
        age = 0

    sleep_hours = quiz.get("sleep_hours", 7)
    try:
        sleep_hours = float(sleep_hours)
    except (ValueError, TypeError):
        sleep_hours = 7

    tipo_pelle = quiz.get("skin_type", "").lower()
    routine = quiz.get("routine_attuale", "").lower()

    # Calibrazione Occhiaie
    if age > 0 and age < 40:
        calibrated["occhiaie"] = min(100, calibrated["occhiaie"] + 15)
    if sleep_hours >= 8:
        calibrated["occhiaie"] = min(100, calibrated["occhiaie"] + 10)
    elif sleep_hours <= 5:
        calibrated["occhiaie"] = max(0, calibrated["occhiaie"] - 10)

    # Calibrazione Disidratazione
    if "secca" in tipo_pelle:
        calibrated["disidratazione"] = max(0, calibrated["disidratazione"] - 15)
    elif "grassa" in tipo_pelle:
        calibrated["disidratazione"] = min(100, calibrated["disidratazione"] + 10)

    # Calibrazione Rughe
    if "completa" in routine or "pro" in routine:
        calibrated["rughe"] = min(100, calibrated["rughe"] + 10)
    if age > 60:
        calibrated["rughe"] = max(0, calibrated["rughe"] - 10)

    # Calibrazione Acne
    if quiz.get("problema_principale", "").lower() in ["acne", "controllo acne"]:
        calibrated["acne"] = max(0, calibrated["acne"] - 10)

    # Ricalcola pelle_pulita_percent
    fields = ["rughe", "pori", "macchie", "occhiaie", "disidratazione", "acne"]
    for field in fields:
        calibrated[field] = max(0, min(100, int(round(calibrated[field]))))
    media = sum(calibrated[f] for f in fields) / len(fields)
    calibrated["pelle_pulita_percent"] = max(0, min(100, int(round(media))))

    return calibrated


# ============================================================
# PROMPTS
# ============================================================

BEAUTY_PROMPT_STANDARD = """Sei un beauty advisor senior (makeup artist con 15 anni di esperienza). Analizza questa foto del viso in close-up per valutare la skin-readiness per routine skincare e makeup.

FOCUS ESTETICO: glow naturale, uniformità, makeup-readiness. NON sei un dermatologo medico.

CHAIN OF THOUGHT:
1. Valuta forma viso, sottotono, età percepita, illuminazione.
2. Identifica beauty concerns (pori visibili? glow naturale? uniformità tono?)
3. Score beauty-readiness 0-100 per ogni parametro.
4. ANCHOR: Occhiaie SOLO se distraggono dal makeup look (ombre naturali da illuminazione NON contano).

SCALA BEAUTY INVERTITA (100=perfetto, 0=problematico):
- Rughe: 100=liscia airbrush-ready, 70=linee sottili normali, 40=rughe visibili coverage needed, 0=deep coverage required
- Pori: 100=invisibili natural finish, 70=appena visibili, 40=visibili primer needed, 0=heavy primer required
- Macchie: 100=tono uniforme no foundation needed, 70=lieve discromia, 40=macchie moderate concealer needed, 0=heavy concealer
- Occhiaie: 100=fresca no corrector needed, 70=lieve ombra normale, 40=occhiaie moderate corrector needed, 0=dark heavy coverage
- Disidratazione: 100=dewy glow naturale, 70=buona idratazione, 40=pelle opaca primer needed, 0=dull dry flaky
- Acne: 100=flawless skin-like, 70=pochi imperfezioni minime, 40=acne moderata green primer, 0=severe active acne
- pelle_pulita_percent: % ready natural makeup (media dei 6 scores sopra)

IMPORTANTE - REGOLE GENERALI DI CALIBRAZIONE:

OCCHIAIE - Regola per età:
- Pelle giovane (20-35): occhiaie sono rare. Solo colorazione blu/viola evidente = score basso. Ombre da illuminazione NON contano. Range tipico: 75-90.
- Pelle media (35-50): lievi occhiaie sono normali. Solo se visibilmente scure/gonfie = score basso. Range tipico: 65-80.
- Pelle matura (50-65): occhiaie moderate sono comuni (pelle più sottile, vasi più visibili). Range tipico: 50-65.
- Pelle anziana (65+): occhiaie moderate-marcate sono fisiologiche. Pelle sottile perioculare mostra vasi. Range tipico: 40-55.

MACCHIE - Regola di severità:
- Pelle con tono quasi uniforme e solo lievissime variazioni naturali di colore (lentiggini sparse, lieve rossore) = 85-95, NON 70-75.
- Score sotto 70 SOLO se ci sono discromie chiaramente visibili (macchie solari definite, melasma, iperpigmentazione post-infiammatoria).
- Lentiggini leggere e uniformi NON sono macchie problematiche.

DISIDRATAZIONE - Regola di valutazione:
- 100=pelle luminosa, dewy, turgida (glow visibile).
- 70=pelle ok ma senza glow particolare, leggermente opaca.
- 50-60=pelle visibilmente opaca, manca luminosità, texture leggermente ruvida.
- Sotto 40=pelle secca evidente, desquamazione, linee di disidratazione.
- Se la pelle appare opaca e senza luminosità ma non secca = 55-65, NON 70+.

PORI - Regola generale:
- Pori poco visibili = 75-85 indipendentemente dall'età.
- Score sotto 60 SOLO se pori chiaramente dilatati e visibili (tipico pelle grassa/mista zona T).

ACNE:
- 100 se non ci sono lesioni attive (comedoni, papule, pustole). Texture normale NON è acne.
- 95 se ci sono 1-2 micro-imperfezioni appena visibili.

ESEMPI BEAUTY ADVISOR (few-shot):

Es1 [donna 40, linee leggere, pelle curata, lieve opacità]:
Valutazione: Pelle discreta, linee sottili normali per età. Tono quasi uniforme con variazioni minime. Pelle leggermente opaca (non dewy). Occhiaie appena percettibili. Pori quasi invisibili.
{"rughe":72,"pori":82,"macchie":88,"occhiaie":75,"disidratazione":60,"acne":95,"pelle_pulita_percent":79}

Es2 [donna 70, rughe profonde, macchie moderate, pelle secca]:
Valutazione: Rughe profonde fronte e naso-labiali. Alcune discromie/macchie solari. Pori poco visibili. Occhiaie moderate (pelle sottile, vasi visibili). Pelle opaca e poco elastica.
{"rughe":20,"pori":80,"macchie":50,"occhiaie":45,"disidratazione":30,"acne":100,"pelle_pulita_percent":54}

Es3 [ragazza 20, acne attiva, pori dilatati]:
Valutazione: Acne attiva su guance. Pori dilatati zona T. Pelle giovane, buona elasticità e idratazione. Tono con segni post-acne. Occhiaie quasi assenti.
{"rughe":95,"pori":35,"macchie":60,"occhiaie":85,"disidratazione":75,"acne":30,"pelle_pulita_percent":63}

Es4 [uomo 45, pelle matura ma curata, lieve disidratazione]:
Valutazione: Rughe moderate fronte e crow feet. Pelle leggermente opaca. Nessuna acne. Tono abbastanza uniforme. Occhiaie lievi normali per età.
{"rughe":55,"pori":75,"macchie":80,"occhiaie":68,"disidratazione":58,"acne":100,"pelle_pulita_percent":73}"""


BEAUTY_PROMPT_DERMOSCOPE = """Sei beauty advisor PRO con dermoscopio. Analizza questa foto DERMOSCOPICA (polarizzata, sub-cutanea).

Questa foto è stata pre-processata con simulazione dermoscopica:
- Cross-polarizzazione: riflessi rimossi, sub-surface visibile
- Unsharp mask: micro-rughe e micro-pori amplificati
- Hue boost: vasi sanguigni e pigmentazione sub-cutanea evidenziati

VEDI COSE INVISIBILI IN SELFIE NORMALI:
- Micro-rughe invisibili a occhio nudo
- Pori reali (non mascherati da makeup/luce)
- Texture sub-skin (disidratazione vera del derma)
- Vasi/occhiaie profonde (non ombre superficiali)

BEAUTY SCALE 0-100 (100=perfetta, 0=problematico):
- Rughe: 100=no micro-lines nemmeno in dermoscopia, 70=micro-rughe normali per età, 40=rughe visibili anche senza dermoscopio, 0=deep dermal damage
- Pori: 100=invisible anche in dermoscopia, 70=micro-pori normali, 40=pori dilatati visibili, 0=large oil control needed
- Macchie: 100=uniform canvas sub-cutaneo, 70=lieve pigmentazione sub-skin, 40=macchie moderate, 0=heavy pigmentation
- Occhiaie: 100=no vascular sub-cutaneo, 70=lieve rete vascolare normale, 40=vasi evidenti corrector needed, 0=deep blue vascular
- Disidratazione: 100=plump dermal water ottimale, 70=buona idratazione dermica, 40=derma opaco, 0=dry cracked matrix
- Acne: 100=flawless sub-cutaneo, 70=micro-comedoni sub-skin, 40=infiammazione sub-cutanea, 0=active inflammatory deep
- pelle_pulita_percent: % ready natural makeup (media dei 6 scores)

IMPORTANTE - DERMOSCOPIA vs SELFIE (REGOLE GENERALI):
- La dermoscopia rivela dettagli nascosti, ma NON inventa problemi che non esistono.
- Scores dermoscopici sono tipicamente 5-10 punti più bassi del selfie per pelle giovane, 10-15 per pelle matura.
- MAI più di 20 punti di differenza rispetto al selfie normale sullo stesso parametro.
- Pori poco visibili in selfie restano relativamente poco visibili in dermoscopia (score 70-80, NON 40-50).
- Micro-rughe e micro-pori visibili SOLO in dermoscopia sono NORMALI e fisiologici: non abbassare troppo lo score.
- Occhiaie: la dermoscopia mostra la rete vascolare, ma vasi lievi sono normali. Stessa logica per età del prompt standard.

ESEMPI DERMOSCOPIA:

Es1 [donna 40, pelle curata, dermoscopia]:
{"rughe":65,"pori":75,"macchie":82,"occhiaie":68,"disidratazione":55,"acne":95,"pelle_pulita_percent":73}

Es2 [donna 70, dermoscopia]:
{"rughe":15,"pori":75,"macchie":42,"occhiaie":40,"disidratazione":25,"acne":95,"pelle_pulita_percent":49}

Es3 [ragazza 20, acne, dermoscopia]:
{"rughe":90,"pori":30,"macchie":55,"occhiaie":80,"disidratazione":68,"acne":25,"pelle_pulita_percent":58}

Valutazione dermoscopica breve, poi JSON esatto:
{"rughe":<0-100>,"pori":<0-100>,"macchie":<0-100>,"occhiaie":<0-100>,"disidratazione":<0-100>,"acne":<0-100>,"pelle_pulita_percent":<0-100>}

Rispondi SOLO con la valutazione breve seguita dal JSON. Nessun altro testo."""


# ============================================================
# HELPER: Chiama Groq e parsa risposta
# ============================================================

def call_groq_vision(img_base64: str, prompt: str, contesto_utente: str = "") -> dict:
    """Chiama Groq Maverick Vision e restituisce scores + ragionamento."""
    full_prompt = prompt
    if contesto_utente:
        full_prompt += f"\n\n{contesto_utente}"
    full_prompt += "\n\nORA ANALIZZA QUESTA FOTO."

    client = get_groq_client()

    response = client.chat.completions.create(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": full_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                ]
            }
        ],
        temperature=0.15,
        max_tokens=800
    )

    result_text = response.choices[0].message.content.strip()

    # Estrai JSON
    json_match = re.search(r'\{[^{}]*"rughe"[^{}]*\}', result_text)
    if not json_match:
        json_match = re.search(r'\{[^{}]+\}', result_text)
    if not json_match:
        raise ValueError(f"Nessun JSON trovato nella risposta: {result_text[:200]}")

    analysis_data = json.loads(json_match.group())

    # Validazione
    required_fields = ["rughe", "pori", "macchie", "occhiaie", "disidratazione", "acne"]
    for field in required_fields:
        if field not in analysis_data:
            analysis_data[field] = 50
        val = analysis_data[field]
        if isinstance(val, (int, float)):
            analysis_data[field] = max(0, min(100, int(round(val))))
        else:
            analysis_data[field] = 50

    # Ricalcola pelle_pulita_percent
    media = sum(analysis_data[f] for f in required_fields) / len(required_fields)
    analysis_data["pelle_pulita_percent"] = max(0, min(100, int(round(media))))

    # Ragionamento
    ragionamento = result_text[:result_text.find('{')].strip() if '{' in result_text else ""

    return {"scores": analysis_data, "ragionamento": ragionamento}


def build_contesto_utente(quiz_data: dict) -> str:
    """Costruisce il contesto utente dal questionario."""
    if not quiz_data:
        return ""
    return f"""
INFORMAZIONI DALL'UTENTE (questionario compilato):
- Età dichiarata: {quiz_data.get('age', 'non specificata')}
- Genere: {quiz_data.get('gender', 'non specificato')}
- Tipo di pelle dichiarato: {quiz_data.get('skin_type', 'non specificato')}
- Problema principale dichiarato: {quiz_data.get('problema_principale', 'non specificato')}
- Routine attuale: {quiz_data.get('routine_attuale', 'non specificata')}
- Ore di sonno: {quiz_data.get('sleep_hours', 'non specificate')}
- Frequenza makeup: {quiz_data.get('makeup_frequency', 'non specificata')}
- Allergie: {quiz_data.get('allergies', 'nessuna')}
- Obiettivo top: {quiz_data.get('obiettivo_top', 'non specificato')}

USA QUESTE INFORMAZIONI per calibrare la tua analisi beauty."""


def build_extras(scores: dict, quiz_data: dict) -> dict:
    """Costruisce armocromia, prodotti, routine, makeup recs."""
    extras = {}

    # Calibrazione
    calibrated = calibrate_beauty_scores(scores, quiz_data)
    extras["calibrated_scores"] = calibrated

    # Armocromia
    extras["armocromia"] = calcola_armocromia(quiz_data)

    # Makeup recommendations
    makeup_recs = []
    if calibrated["pori"] < 60:
        makeup_recs.append("Primer pori (silicone-based)")
    if calibrated["occhiaie"] < 60:
        makeup_recs.append("Corrector occhiaie (pesca/arancio)")
    if calibrated["macchie"] < 60:
        makeup_recs.append("Concealer alta copertura")
    if calibrated["disidratazione"] < 60:
        makeup_recs.append("Dewy foundation + setting spray idratante")
    elif calibrated["disidratazione"] >= 70:
        makeup_recs.append("Foundation naturale o tinted moisturizer")
    if calibrated["acne"] < 60:
        makeup_recs.append("Green primer + fondotinta oil-free")
    if calibrated["rughe"] < 60:
        makeup_recs.append("Primer anti-age + foundation idratante (no matte)")
    if not makeup_recs:
        makeup_recs.append("Minimal makeup: BB cream + mascara")
    extras["makeup_recommendations"] = makeup_recs

    # Skincare routine recs
    skincare_recs = []
    if calibrated["disidratazione"] < 70:
        skincare_recs.append("Siero acido ialuronico (mattina)")
    if calibrated["macchie"] < 70:
        skincare_recs.append("Siero vitamina C (mattina)")
    if calibrated["rughe"] < 60:
        skincare_recs.append("Retinolo 0.3% (sera, 2-3x/settimana)")
    if calibrated["acne"] < 70:
        skincare_recs.append("Niacinamide 10% (sera)")
    if calibrated["pori"] < 60:
        skincare_recs.append("BHA/Acido salicilico (sera, 2x/settimana)")
    skincare_recs.append("SPF 50+ (sempre, ogni mattina)")
    extras["skincare_routine"] = skincare_recs

    # Link prodotti Amazon
    tipo_pelle = quiz_data.get("skin_type", "Mista")
    problema = quiz_data.get("problema_principale", "")
    obiettivo = quiz_data.get("obiettivo_top", "")
    if not problema:
        score_map = {
            "Rughe": calibrated["rughe"],
            "Pori/Macchie": min(calibrated["pori"], calibrated["macchie"]),
            "Occhiaie": calibrated["occhiaie"],
            "Disidratazione": calibrated["disidratazione"],
            "Acne": calibrated["acne"]
        }
        problema = min(score_map, key=score_map.get)
    extras["prodotti_consigliati"] = genera_link_amazon(problema, tipo_pelle, obiettivo)

    # Routine completa
    extras["routine"] = genera_routine(calibrated, quiz_data)

    return extras


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/health")
async def health():
    return {"status": "ok", "engine": "Groq Maverick + Dermoscopio Virtuale", "version": "3.1"}


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    questionnaire: str = Form(default="{}")
):
    """
    Analisi Beauty Standard: foto normale + Maverick beauty advisor.
    Scala INVERTITA: 100=perfetto, 0=problematico.
    """
    try:
        try:
            quiz_data = json.loads(questionnaire)
        except json.JSONDecodeError:
            quiz_data = {}

        img_bytes = await file.read()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        contesto = build_contesto_utente(quiz_data)
        groq_result = call_groq_vision(img_base64, BEAUTY_PROMPT_STANDARD, contesto)

        result = {
            "status": "success",
            "mode": "standard",
            "beauty_scores": groq_result["scores"],
            "ragionamento": groq_result["ragionamento"]
        }

        if quiz_data and len(quiz_data) > 2:
            extras = build_extras(groq_result["scores"], quiz_data)
            result.update(extras)
        else:
            result["calibrated_scores"] = groq_result["scores"]

        return result

    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/analyze-dermoscope")
async def analyze_dermoscope(
    file: UploadFile = File(...),
    questionnaire: str = Form(default="{}")
):
    """
    Analisi Dermoscopio PRO: pre-processing OpenCV + Maverick beauty advisor.
    Rivela micro-rughe, pori reali, texture sub-cutanea.
    Scala INVERTITA: 100=perfetto, 0=problematico.
    """
    try:
        try:
            quiz_data = json.loads(questionnaire)
        except json.JSONDecodeError:
            quiz_data = {}

        # 1. Leggi immagine originale
        img_bytes = await file.read()

        # 2. Applica dermoscopio virtuale
        derm_bytes, derm_b64 = dermoscope_effect(img_bytes)

        # 3. Analizza immagine dermoscopica con Maverick
        derm_base64 = base64.b64encode(derm_bytes).decode("utf-8")
        contesto = build_contesto_utente(quiz_data)
        groq_result = call_groq_vision(derm_base64, BEAUTY_PROMPT_DERMOSCOPE, contesto)

        result = {
            "status": "success",
            "mode": "dermoscope",
            "beauty_scores": groq_result["scores"],
            "ragionamento": groq_result["ragionamento"],
            "derm_image_b64": derm_b64,
            "processing": "dermoscope_virtual_applied"
        }

        if quiz_data and len(quiz_data) > 2:
            extras = build_extras(groq_result["scores"], quiz_data)
            result.update(extras)
        else:
            result["calibrated_scores"] = groq_result["scores"]

        return result

    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
