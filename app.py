from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import base64
import json
import re
from groq import Groq

app = FastAPI()

# Configurazione CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inizializza client Groq
groq_client = None

def get_groq_client():
    global groq_client
    if groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="GROQ_API_KEY non configurata")
        groq_client = Groq(api_key=api_key)
    return groq_client

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        # Leggi l'immagine
        img_bytes = await file.read()
        
        # Converti in base64
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        # Prompt few-shot + Chain-of-Thought con scale dermatologiche
        prompt = """Sei un dermatologo con 20 anni di esperienza clinica. Analizza questa foto del viso close-up.

PROCEDURA (Chain-of-Thought):
1. Identifica eta approssimativa, etnia probabile e condizioni di illuminazione della foto.
2. Valuta ogni feature della pelle usando le scale dermatologiche standard sotto.
3. Assegna uno score da 0 a 100 per ogni parametro.
4. Calcola pelle_pulita_percent come: 100 - (media di rughe + pori + macchie + acne + occhiaie + disidratazione) / 6
5. Restituisci SOLO il JSON finale.

SCALE DERMATOLOGICHE DI RIFERIMENTO:
- rughe: scala Glogau (0=nessuna ruga visibile, 20=linee sottili minime, 40=rughe dinamiche moderate, 60=rughe a riposo visibili, 80=solchi profondi permanenti, 100=rughe profonde Glogau IV)
- pori: (0=invisibili ad occhio nudo, 20=appena percettibili, 40=visibili zona T, 60=dilatati evidenti, 80=molto dilatati e aperti, 100=pori larghi e oleosi su tutto il viso)
- macchie: scala MASI-like (0=tono uniforme senza discromie, 20=rare lentiggini leggere, 40=alcune macchie localizzate, 60=discromie moderate diffuse, 80=iperpigmentazione estesa, 100=melasma/discromie severe diffuse)
- occhiaie: (0=nessuna ombra perioculare, 15=ombra lievissima appena percettibile, 30=occhiaie leggere visibili, 50=occhiaie moderate con colorazione evidente, 70=occhiaie marcate scure, 100=occhiaie profonde vascolari molto scure)
- disidratazione: (0=pelle turgida luminosa idratata, 20=leggera secchezza appena percettibile, 40=secchezza moderata con zone opache, 60=pelle visibilmente secca e opaca, 80=pelle molto secca poco elastica, 100=pelle disidratata desquamata screpolata)
- acne: scala Leeds (0=pelle completamente priva di imperfezioni, 10=1-2 micro-comedoni appena visibili, 30=comedoni sparsi e qualche papula, 50=acne moderata papulo-pustolosa, 70=acne diffusa con pustole, 100=acne severa nodulare/cistica)
- pelle_pulita_percent: calcolato come 100 - (rughe + pori + macchie + occhiaie + disidratazione + acne) / 6

ESEMPI DI RIFERIMENTO (few-shot):

Esempio 1 - Donna 25 anni, asiatica, buona illuminazione, pelle liscia:
Ragionamento: Pelle giovane con texture fine. Linee minime naso-labiali appena accennate. Pori quasi invisibili. Nessuna macchia o acne. Pelle ben idratata e luminosa. Occhiaie assenti.
{"rughe":10,"pori":15,"macchie":5,"occhiaie":8,"disidratazione":10,"acne":0,"pelle_pulita_percent":92}

Esempio 2 - Uomo 45 anni, caucasico, luce naturale, rughe fronte moderate:
Ragionamento: Zampe di gallina moderate ai lati degli occhi. Rughe orizzontali sulla fronte visibili. Pori leggermente dilatati nella zona T. Qualche macchia solare sugli zigomi. Pelle leggermente disidratata. Occhiaie moderate.
{"rughe":50,"pori":35,"macchie":25,"occhiaie":35,"disidratazione":40,"acne":0,"pelle_pulita_percent":69}

Esempio 3 - Donna 65 anni, pelle chiara, macchie solari evidenti:
Ragionamento: Rughe profonde sulla fronte e solchi naso-labiali marcati. Pori poco visibili nonostante l'eta. Diverse macchie solari e lentigo sulle guance. Pelle opaca e poco elastica, disidratazione evidente. Occhiaie moderate con componente vascolare. Nessuna acne.
{"rughe":75,"pori":20,"macchie":60,"occhiaie":45,"disidratazione":65,"acne":0,"pelle_pulita_percent":56}

Esempio 4 - Ragazzo 19 anni, pelle mista, acne attiva con pori dilatati:
Ragionamento: Pelle giovane senza rughe. Pori molto dilatati su guance e naso. Segni post-acne con macchie residue. Comedoni e pustole attive su fronte e guance. Pelle mista con zone oleose. Occhiaie leggere.
{"rughe":5,"pori":65,"macchie":35,"occhiaie":20,"disidratazione":15,"acne":60,"pelle_pulita_percent":67}

ISTRUZIONI FINALI:
Per la foto fornita, scrivi un breve ragionamento (2-3 frasi) e poi restituisci SOLO il JSON.
Il JSON deve essere l'ULTIMA cosa nella tua risposta, su una riga sola.
Formato esatto: {"rughe":X,"pori":X,"macchie":X,"occhiaie":X,"disidratazione":X,"acne":X,"pelle_pulita_percent":X}"""

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
        
        # Cerca il JSON nella risposta (potrebbe avere ragionamento prima)
        json_str = None
        
        # Metodo 1: Cerca l'ultimo oggetto JSON valido nella risposta
        json_matches = re.findall(r'\{[^{}]*\}', result_text)
        if json_matches:
            for match in reversed(json_matches):
                try:
                    parsed = json.loads(match)
                    if "rughe" in parsed and "pori" in parsed:
                        json_str = match
                        break
                except json.JSONDecodeError:
                    continue
        
        # Metodo 2: Fallback - pulisci markdown
        if json_str is None:
            clean_text = result_text
            if clean_text.startswith("```json"):
                clean_text = clean_text.replace("```json", "").replace("```", "").strip()
            elif clean_text.startswith("```"):
                clean_text = clean_text.replace("```", "").strip()
            json_str = clean_text
        
        # Parse JSON
        analysis_data = json.loads(json_str)
        
        # Verifica che tutti i campi richiesti siano presenti
        required_fields = ["rughe", "pori", "macchie", "occhiaie", "disidratazione", "acne", "pelle_pulita_percent"]
        for field in required_fields:
            if field not in analysis_data:
                analysis_data[field] = 50
        
        # Assicura che i valori siano interi tra 0 e 100
        for field in required_fields:
            val = analysis_data[field]
            if isinstance(val, (int, float)):
                analysis_data[field] = max(0, min(100, int(round(val))))
            else:
                analysis_data[field] = 50
        
        # Ricalcola pelle_pulita_percent per coerenza
        avg_issues = (
            analysis_data["rughe"] + 
            analysis_data["pori"] + 
            analysis_data["macchie"] + 
            analysis_data["occhiaie"] + 
            analysis_data["disidratazione"] + 
            analysis_data["acne"]
        ) / 6
        analysis_data["pelle_pulita_percent"] = max(0, min(100, int(round(100 - avg_issues))))
        
        # Restituisci solo i campi richiesti
        return {field: analysis_data[field] for field in required_fields}
        
    except json.JSONDecodeError as e:
        return {
            "rughe": 50,
            "pori": 50,
            "macchie": 50,
            "occhiaie": 50,
            "disidratazione": 50,
            "acne": 50,
            "pelle_pulita_percent": 50,
            "error": "Errore nel parsing della risposta AI"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore durante l'analisi: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
