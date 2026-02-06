from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import base64
import json
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
        
        # Prepara il prompt per l'analisi della pelle
        prompt = """Analizza questa foto del viso e fornisci un'analisi dettagliata della pelle.
Restituisci SOLO un oggetto JSON valido con questa struttura esatta (valori da 0 a 100):
{
  "rughe": <numero 0-100>,
  "pori": <numero 0-100>,
  "macchie": <numero 0-100>,
  "occhiaie": <numero 0-100>,
  "glow": <numero 0-100>,
  "acne": <numero 0-100>,
  "pelle_pulita_percent": <numero 0-100>
}

Dove:
- rughe: livello di rughe visibili (0=nessuna, 100=molte)
- pori: visibilità dei pori (0=invisibili, 100=molto visibili)
- macchie: presenza di macchie/iperpigmentazione (0=nessuna, 100=molte)
- occhiaie: intensità delle occhiaie (0=nessuna, 100=molto scure)
- glow: luminosità della pelle (0=opaca, 100=molto luminosa)
- acne: presenza di acne/imperfezioni (0=nessuna, 100=severa)
- pelle_pulita_percent: percentuale generale di pulizia della pelle (0=pessima, 100=perfetta)

Rispondi SOLO con il JSON, senza testo aggiuntivo."""

        # Chiama Groq Vision API
        client = get_groq_client()
        
        response = client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
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
            temperature=0.3,
            max_tokens=500
        )
        
        # Estrai la risposta
        result_text = response.choices[0].message.content.strip()
        
        # Pulisci la risposta da eventuali markdown
        if result_text.startswith("```json"):
            result_text = result_text.replace("```json", "").replace("```", "").strip()
        elif result_text.startswith("```"):
            result_text = result_text.replace("```", "").strip()
        
        # Parse JSON
        analysis_data = json.loads(result_text)
        
        # Verifica che tutti i campi richiesti siano presenti
        required_fields = ["rughe", "pori", "macchie", "occhiaie", "glow", "acne", "pelle_pulita_percent"]
        for field in required_fields:
            if field not in analysis_data:
                analysis_data[field] = 50  # Valore di default
        
        return analysis_data
        
    except json.JSONDecodeError as e:
        return {
            "rughe": 50,
            "pori": 50,
            "macchie": 50,
            "occhiaie": 50,
            "glow": 50,
            "acne": 50,
            "pelle_pulita_percent": 50,
            "error": "Errore nel parsing della risposta AI"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore durante l'analisi: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
