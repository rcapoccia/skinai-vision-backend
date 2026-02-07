from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import base64
import io
import re
import json
import os

app = FastAPI(title="SkinGlow AI Vision Backend")

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
# ARMOCROMIA ENGINE
# ============================================================

def calcola_armocromia(questionnaire: dict) -> dict:
    """
    Calcola la stagione armocromia basata sulle risposte del questionario.
    Usa: colore capelli, colore occhi, vene polso, abbronzatura.
    Restituisce: stagione, sottotono, palette colori, consigli makeup.
    """
    punti_caldo = 0
    punti_freddo = 0
    intensita_alta = 0
    intensita_bassa = 0

    # Vene polso (indicatore più affidabile)
    vene = questionnaire.get("vene_polso", "")
    if "Verdi" in vene:
        punti_caldo += 3
    elif "Blu" in vene:
        punti_freddo += 3
    else:
        punti_caldo += 1
        punti_freddo += 1

    # Abbronzatura
    abbronzatura = questionnaire.get("abbronzatura", "")
    if "dorata" in abbronzatura.lower() or "facile" in abbronzatura.lower():
        punti_caldo += 2
    elif "rossore" in abbronzatura.lower() or "difficile" in abbronzatura.lower():
        punti_freddo += 2
    elif "non mi" in abbronzatura.lower():
        punti_freddo += 1

    # Colore capelli
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

    # Colore occhi
    occhi = questionnaire.get("eye_color", "").lower()
    if any(c in occhi for c in ["marrone scuro", "nero"]):
        intensita_alta += 1
        punti_caldo += 1
    elif any(c in occhi for c in ["blu", "celeste", "grigio"]):
        punti_freddo += 1
        intensita_bassa += 1
    elif any(c in occhi for c in ["verde", "nocciola", "marrone chiaro"]):
        punti_caldo += 1

    # Determina sottotono
    sottotono = "Caldo" if punti_caldo > punti_freddo else "Freddo" if punti_freddo > punti_caldo else "Neutro"

    # Determina stagione
    if sottotono == "Caldo":
        if intensita_alta >= intensita_bassa:
            stagione = "Autunno"
            sotto_stagione = "Autunno Caldo"
            palette = ["Terracotta", "Arancio bruciato", "Oro antico", "Verde oliva", "Marrone cioccolato", "Senape"]
            makeup = {
                "fondotinta": "Sottotono dorato/pesca",
                "blush": "Pesca, terracotta",
                "labbra": "Nude caldo, mattone, arancio bruciato",
                "occhi": "Bronzo, rame, verde oliva, marrone caldo"
            }
        else:
            stagione = "Primavera"
            sotto_stagione = "Primavera Calda"
            palette = ["Pesca", "Corallo", "Oro chiaro", "Verde mela", "Azzurro caldo", "Avorio"]
            makeup = {
                "fondotinta": "Sottotono pesca/dorato chiaro",
                "blush": "Pesca chiaro, corallo",
                "labbra": "Corallo, pesca, rosa caldo",
                "occhi": "Pesca, oro chiaro, verde chiaro, marrone chiaro"
            }
    else:
        if intensita_alta >= intensita_bassa:
            stagione = "Inverno"
            sotto_stagione = "Inverno Freddo"
            palette = ["Nero", "Bianco puro", "Fucsia", "Blu royal", "Rosso ciliegia", "Argento"]
            makeup = {
                "fondotinta": "Sottotono rosa/neutro freddo",
                "blush": "Rosa freddo, berry",
                "labbra": "Rosso ciliegia, fucsia, berry, rosa freddo",
                "occhi": "Grigio, argento, blu navy, prugna"
            }
        else:
            stagione = "Estate"
            sotto_stagione = "Estate Fredda"
            palette = ["Rosa antico", "Lavanda", "Grigio perla", "Azzurro polvere", "Malva", "Argento chiaro"]
            makeup = {
                "fondotinta": "Sottotono rosa/beige freddo",
                "blush": "Rosa antico, malva",
                "labbra": "Rosa antico, malva, berry chiaro",
                "occhi": "Grigio, lavanda, rosa antico, taupe"
            }

    return {
        "stagione": stagione,
        "sotto_stagione": sotto_stagione,
        "sottotono": sottotono,
        "palette_colori": palette,
        "consigli_makeup": makeup
    }


# ============================================================
# LINK PRODOTTI AMAZON DINAMICI
# ============================================================

PRINCIPI_ATTIVI = {
    "Disidratazione": {
        "attivo": "Acido Ialuronico",
        "query_base": "acido ialuronico siero"
    },
    "Rughe": {
        "attivo": "Retinolo",
        "query_base": "retinolo crema anti rughe"
    },
    "Acne": {
        "attivo": "Niacinamide",
        "query_base": "niacinamide siero acne"
    },
    "Pori/Macchie": {
        "attivo": "Vitamina C",
        "query_base": "vitamina c siero antimacchie"
    },
    "Occhiaie": {
        "attivo": "Caffeina + Vitamina K",
        "query_base": "contorno occhi caffeina occhiaie"
    },
    "Sensibilità": {
        "attivo": "Ceramidi + Centella",
        "query_base": "crema lenitiva ceramidi pelle sensibile"
    },
    "Luminosità": {
        "attivo": "Vitamina C + AHA",
        "query_base": "vitamina c siero luminosita"
    },
    "Antietà": {
        "attivo": "Retinolo + Peptidi",
        "query_base": "retinolo peptidi anti age"
    },
    "Idratazione": {
        "attivo": "Acido Ialuronico + Ceramidi",
        "query_base": "acido ialuronico ceramidi idratante"
    },
    "Controllo acne": {
        "attivo": "Acido Salicilico + Niacinamide",
        "query_base": "acido salicilico niacinamide acne"
    },
    "Uniformità tono": {
        "attivo": "Vitamina C + Niacinamide",
        "query_base": "vitamina c niacinamide uniformante"
    }
}

def genera_link_amazon(problema: str, tipo_pelle: str, obiettivo: str = "") -> list:
    """Genera link Amazon dinamici basati su problema, tipo pelle e obiettivo."""
    prodotti = []

    # Prodotto per il problema principale
    if problema in PRINCIPI_ATTIVI:
        info = PRINCIPI_ATTIVI[problema]
        query = f"{info['query_base']} pelle {tipo_pelle.lower()}"
        prodotti.append({
            "problema": problema,
            "principio_attivo": info["attivo"],
            "query": query,
            "link_amazon": f"https://www.amazon.it/s?k={query.replace(' ', '+')}"
        })

    # Prodotto per l'obiettivo top
    if obiettivo and obiettivo in PRINCIPI_ATTIVI and obiettivo != problema:
        info = PRINCIPI_ATTIVI[obiettivo]
        query = f"{info['query_base']} pelle {tipo_pelle.lower()}"
        prodotti.append({
            "problema": obiettivo,
            "principio_attivo": info["attivo"],
            "query": query,
            "link_amazon": f"https://www.amazon.it/s?k={query.replace(' ', '+')}"
        })

    # Sempre consiglia SPF
    spf_query = f"crema solare viso SPF50 pelle {tipo_pelle.lower()}"
    prodotti.append({
        "problema": "Protezione solare (essenziale)",
        "principio_attivo": "SPF 50+",
        "query": spf_query,
        "link_amazon": f"https://www.amazon.it/s?k={spf_query.replace(' ', '+')}"
    })

    return prodotti


# ============================================================
# ROUTINE PERSONALIZZATA
# ============================================================

def genera_routine(scores: dict, questionnaire: dict) -> dict:
    """Genera routine skincare personalizzata basata su scores e questionario."""
    routine_attuale = questionnaire.get("routine_attuale", "Nessuna")
    tipo_pelle = questionnaire.get("skin_type", "Mista")
    problema = questionnaire.get("problema_principale", "")

    # Routine mattina
    mattina = ["Detergente delicato"]

    if scores.get("disidratazione", 0) > 40 or tipo_pelle in ["Secca", "Sensibile"]:
        mattina.append("Siero acido ialuronico")
    if scores.get("macchie", 0) > 30 or problema == "Luminosità":
        mattina.append("Siero vitamina C")
    mattina.append("Crema idratante")
    mattina.append("SPF 50 (sempre!)")

    # Routine sera
    sera = []
    if questionnaire.get("makeup_frequency", "") in ["Quotidiano leggero", "Quotidiano completo"]:
        sera.append("Doppia detersione (olio + gel)")
    else:
        sera.append("Detergente delicato")

    if scores.get("acne", 0) > 30 or tipo_pelle == "Grassa":
        sera.append("Tonico con niacinamide")
    if scores.get("rughe", 0) > 40 and routine_attuale in ["Completa (sieri+SPF)", "Pro (retinolo/acidi)"]:
        sera.append("Retinolo (2-3 volte/settimana)")
    elif scores.get("rughe", 0) > 40:
        sera.append("Siero peptidi anti-age")
    if scores.get("occhiaie", 0) > 30:
        sera.append("Contorno occhi con caffeina")
    sera.append("Crema notte nutriente")

    # Settimanale
    settimanale = []
    if scores.get("pori", 0) > 30 or tipo_pelle in ["Grassa", "Mista"]:
        settimanale.append("Maschera argilla (1x/settimana)")
    if scores.get("disidratazione", 0) > 40:
        settimanale.append("Maschera idratante (2x/settimana)")
    if routine_attuale in ["Completa (sieri+SPF)", "Pro (retinolo/acidi)"]:
        settimanale.append("Esfoliante AHA/BHA (1-2x/settimana)")

    return {
        "mattina": mattina,
        "sera": sera,
        "settimanale": settimanale,
        "nota": f"Routine calibrata per pelle {tipo_pelle.lower()}, livello {routine_attuale.lower()}"
    }


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/health")
async def health():
    return {"status": "ok", "engine": "Groq Vision + Armocromia", "version": "2.0"}


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    questionnaire: str = Form(default="{}")
):
    """
    Analisi combinata: foto + questionario.
    - file: foto del viso (JPG/PNG)
    - questionnaire: JSON string con le risposte del questionario (opzionale)
    """
    try:
        # Parse questionario
        try:
            quiz_data = json.loads(questionnaire)
        except json.JSONDecodeError:
            quiz_data = {}

        # Leggi e codifica l'immagine
        img_bytes = await file.read()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        # Costruisci il contesto dal questionario
        contesto_utente = ""
        if quiz_data:
            contesto_utente = f"""
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

USA QUESTE INFORMAZIONI per calibrare la tua analisi. Ad esempio:
- Se l'utente dichiara pelle secca, dai più peso alla disidratazione.
- Se l'utente dorme poco (<6h), le occhiaie potrebbero essere più marcate.
- L'età dichiarata aiuta a contestualizzare le rughe (normali per l'età vs premature).
"""

        # Prompt con few-shot + CoT + scale dermatologiche
        prompt = f"""Sei un dermatologo con 20 anni di esperienza clinica. Analizza questa foto del viso in close-up.

RAGIONA PASSO-PASSO:
1. Identifica età approssimativa, etnia probabile e condizioni di illuminazione della foto.
2. Valuta ogni feature usando le scale dermatologiche standard sotto.
3. Assegna uno score da 0 a 100 per ogni parametro.
4. Restituisci SOLO il JSON finale.

SCALE DERMATOLOGICHE DI RIFERIMENTO:
- Rughe: scala Glogau (0=nessuna linea visibile Glogau I, 25=linee minime a riposo Glogau II, 50=rughe visibili a riposo Glogau III, 75-100=rughe profonde diffuse Glogau IV)
- Pori: 0=invisibili ad occhio nudo, 30=leggermente visibili zona T, 60=visibili e dilatati, 100=molto dilatati e oleosi
- Macchie: scala MASI-like (0=tono uniforme, 25=lieve discromia, 50=macchie moderate, 75-100=iperpigmentazione estesa)
- Occhiaie: 0=nessuna ombra periorbitale, 25=lieve ombra, 50=occhiaie moderate vascolari, 75-100=occhiaie profonde scure
- Disidratazione: 0=pelle turgida e luminosa, 25=lieve secchezza, 50=pelle opaca e poco elastica, 75-100=pelle secca desquamata
- Acne: scala Leeds (0=nessuna lesione, 25=pochi comedoni, 50=papule e pustole moderate, 75-100=acne nodulare severa)
- Pelle pulita %: calcolata come 100 - (media di rughe+pori+macchie+occhiaie+disidratazione+acne)/6

{contesto_utente}

ESEMPI DI RIFERIMENTO (few-shot):

Esempio 1 - Donna 25 anni, asiatica, buona illuminazione, pelle liscia:
Ragionamento: Pelle giovane con texture fine. Linee minime naso-labiali appena percettibili. Pori quasi invisibili. Tono uniforme. Lieve disidratazione zona guance. Nessuna lesione acneica.
{{"rughe":15,"pori":20,"macchie":5,"occhiaie":10,"disidratazione":20,"acne":0,"pelle_pulita_percent":88}}

Esempio 2 - Uomo 45 anni, caucasico, luce naturale, rughe fronte moderate:
Ragionamento: Rughe di espressione fronte e zampe di gallina moderate. Pori visibili zona T. Qualche macchia solare guance. Occhiaie moderate. Pelle leggermente disidratata. Nessuna acne attiva.
{{"rughe":50,"pori":35,"macchie":20,"occhiaie":35,"disidratazione":40,"acne":0,"pelle_pulita_percent":55}}

Esempio 3 - Donna 65 anni, pelle chiara, macchie solari evidenti:
Ragionamento: Rughe profonde fronte e solchi naso-labiali. Pori poco visibili. Discromie e macchie solari diffuse. Occhiaie moderate. Pelle opaca e poco elastica. Nessuna acne.
{{"rughe":75,"pori":20,"macchie":55,"occhiaie":45,"disidratazione":65,"acne":0,"pelle_pulita_percent":27}}

Esempio 4 - Ragazzo 19 anni, acne attiva, pori dilatati:
Ragionamento: Pelle giovane ma con acne attiva. Comedoni e pustole su guance e fronte. Pori molto dilatati zona T. Qualche segno post-acneico. Pelle grassa non disidratata. Occhiaie minime.
{{"rughe":5,"pori":65,"macchie":35,"occhiaie":15,"disidratazione":15,"acne":60,"pelle_pulita_percent":35}}

ORA ANALIZZA QUESTA FOTO.
Ragionamento breve, poi JSON esatto con questa struttura:
{{"rughe":<0-100>,"pori":<0-100>,"macchie":<0-100>,"occhiaie":<0-100>,"disidratazione":<0-100>,"acne":<0-100>,"pelle_pulita_percent":<0-100>}}

Rispondi SOLO con il ragionamento breve seguito dal JSON. Nessun altro testo."""

        # Chiama Groq Vision API
        client = get_groq_client()

        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.2,
            max_tokens=800
        )

        # Estrai la risposta
        result_text = response.choices[0].message.content.strip()

        # Estrai il JSON dalla risposta (può essere dopo il ragionamento CoT)
        json_match = re.search(r'\{[^{}]*"rughe"[^{}]*\}', result_text)
        if json_match:
            json_str = json_match.group()
        else:
            # Prova a trovare qualsiasi JSON valido
            json_match = re.search(r'\{[^{}]+\}', result_text)
            if json_match:
                json_str = json_match.group()
            else:
                return {"status": "error", "message": "Nessun JSON trovato nella risposta", "raw": result_text}

        # Parse JSON
        analysis_data = json.loads(json_str)

        # Validazione e normalizzazione valori (0-100 interi)
        required_fields = ["rughe", "pori", "macchie", "occhiaie", "disidratazione", "acne"]
        for field in required_fields:
            if field not in analysis_data:
                analysis_data[field] = 0
            val = analysis_data[field]
            if isinstance(val, (int, float)):
                analysis_data[field] = max(0, min(100, int(round(val))))
            else:
                analysis_data[field] = 0

        # Ricalcola pelle_pulita_percent con formula
        media_problemi = sum(analysis_data[f] for f in required_fields) / len(required_fields)
        analysis_data["pelle_pulita_percent"] = max(0, min(100, int(round(100 - media_problemi))))

        # Estrai ragionamento CoT (tutto prima del JSON)
        ragionamento = result_text[:result_text.find('{')].strip() if '{' in result_text else ""

        # Costruisci risposta completa
        result = {
            "status": "success",
            "scores": analysis_data,
            "ragionamento": ragionamento
        }

        # Se c'è il questionario, aggiungi armocromia, prodotti e routine
        if quiz_data and len(quiz_data) > 2:
            # Armocromia
            armocromia = calcola_armocromia(quiz_data)
            result["armocromia"] = armocromia

            # Link prodotti Amazon
            tipo_pelle = quiz_data.get("skin_type", "Mista")
            problema = quiz_data.get("problema_principale", "")
            obiettivo = quiz_data.get("obiettivo_top", "")

            # Se non c'è problema dichiarato, usa il punteggio più alto
            if not problema:
                score_map = {
                    "Rughe": analysis_data["rughe"],
                    "Pori/Macchie": max(analysis_data["pori"], analysis_data["macchie"]),
                    "Occhiaie": analysis_data["occhiaie"],
                    "Disidratazione": analysis_data["disidratazione"],
                    "Acne": analysis_data["acne"]
                }
                problema = max(score_map, key=score_map.get)

            prodotti = genera_link_amazon(problema, tipo_pelle, obiettivo)
            result["prodotti_consigliati"] = prodotti

            # Routine personalizzata
            routine = genera_routine(analysis_data, quiz_data)
            result["routine"] = routine

        return result

    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
