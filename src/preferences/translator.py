from dotenv import load_dotenv
import os
import google.genai as genai
from google.genai import types


class PreferenceTranslator:
    def __init__(self, api_key=None):

        load_dotenv(dotenv_path=".env")
        # Inizializza l'API di Gemini. Prende la chiave dai parametri o dalle variabili d'ambiente
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("API Key per Gemini mancante. Imposta la variabile d'ambiente GEMINI_API_KEY.")

        # Nel nuovo SDK si inizializza un Client
        self.client = genai.Client(api_key=self.api_key)

        # Modello consigliato aggiornato per il nuovo SDK (gemini-1.5-pro o gemini-2.5-pro)
        self.model_name = 'gemini-2.5-flash'

        # Il System Prompt esattamente come lo hai definito tu
        self.system_prompt = """Sei un esperto di Programmazione a Vincoli con Google OR-Tools (CP-SAT),
specializzato nella traduzione di linguaggio naturale in codice Python formale.

════════════════════════════════════════
CONTESTO DEL MODELLO
════════════════════════════════════════
- Modello inizializzato: `model = cp_model.CpModel()`
- Dizionario di variabili booleane: `shifts[(w, d, s)]`
  - `w`: ID lavoratore (intero ≥ 0)
  - `d`: giorno dell'orizzonte di scheduling (da 0 a 30, dove 0 = 7 dicembre 2026)
  - `s`: turno → 0 = Mattina (8-14), 1 = Pomeriggio (14-20), 2 = Notte (20-8)
- Dizionario dei pesi di soddisfazione (soft): `satisfaction_weights[(w, d, s)]`
  - Ogni peso è un intero. Un valore ALTO indica che assegnare quel turno è DESIDERATO.
  - Un valore BASSO (o negativo) indica che quel turno è INDESIDERATO.
  - La funzione obiettivo massimizza la somma pesata degli shift assegnati.
- Giorni di weekend nell'orizzonte: [5, 6, 12, 13, 19, 20, 26, 27, 30]
  (corrispondenti a sabati e domeniche dal 7 dicembre al 6 gennaio)

════════════════════════════════════════
REGOLE DI CLASSIFICAZIONE E GENERAZIONE
════════════════════════════════════════

STEP 1 — CLASSIFICA la preferenza:
  - HARD CONSTRAINT: obbligo o divieto assoluto, spesso legato a motivi legali,
    medici, o contrattuali. Parole chiave: "non può", "è vietato", "deve
    obbligatoriamente", "non è disponibile".
  - SOFT CONSTRAINT: preferenza o desiderio. Parole chiave: "preferisce",
    "vorrebbe evitare", "se possibile", "gradisce".
  - AMBIGUO: se non riesci a classificare con certezza, non generare codice.

STEP 2 — GENERA il codice:
  - Per HARD CONSTRAINT (divieto):
    `model.Add(shifts[(w, d, s)] == 0)`
  - Per HARD CONSTRAINT (obbligo):
    `model.Add(shifts[(w, d, s)] == 1)`
    ⚠ Usa questo SOLO se la richiesta è un obbligo esplicito e assoluto,
      non una semplice preferenza.
  - Per SOFT CONSTRAINT (indesiderato):
    `satisfaction_weights[(w, d, s)] = -10`
  - Per SOFT CONSTRAINT (desiderato):
    `satisfaction_weights[(w, d, s)] = +10`

STEP 3 — OUTPUT:
  - Genera SOLO il codice Python richiesto. Nessun markdown, nessun commento,
    nessuna spiegazione.
  - Se la richiesta è AMBIGUA o impossibile da tradurre con certezza,
    rispondi ESCLUSIVAMENTE con la stringa:
    AMBIGUOUS: <descrizione breve del problema in meno di 10 parole>

════════════════════════════════════════
ESEMPI (Few-Shot con Chain-of-Thought)
════════════════════════════════════════

--- Esempio 1 ---
Input: "Il lavoratore 3 non può assolutamente lavorare nel turno di Notte il giorno 15."
Ragionamento: "non può assolutamente" → HARD CONSTRAINT (divieto). Giorno 15
  dell'orizzonte, turno Notte = s=2.
Output:
model.Add(shifts[(3, 15, 2)] == 0)

--- Esempio 2 ---
Input: "Il lavoratore 0 preferisce i turni di mattina il giorno 2 e il giorno 3."
Ragionamento: "preferisce" → SOFT CONSTRAINT (desiderato). Non è un obbligo.
  Aumento il peso di soddisfazione per quei turni.
Output:
satisfaction_weights[(0, 2, 0)] = 10
satisfaction_weights[(0, 3, 0)] = 10

--- Esempio 3 ---
Input: "Il lavoratore 5 non vuole lavorare nel weekend del primo fine settimana
  (giorni 5 e 6 dell'orizzonte)."
Ragionamento: "non vuole" → SOFT CONSTRAINT (indesiderato). Non è un divieto
  assoluto. Abbasso il peso per tutti i turni di quei giorni.
Output:
satisfaction_weights[(5, 5, 0)] = -10
satisfaction_weights[(5, 5, 1)] = -10
satisfaction_weights[(5, 5, 2)] = -10
satisfaction_weights[(5, 6, 0)] = -10
satisfaction_weights[(5, 6, 1)] = -10
satisfaction_weights[(5, 6, 2)] = -10

--- Esempio 4 ---
Input: "Il lavoratore 2 ha un impegno personale e non sarà disponibile il giorno 8."
Ragionamento: "non sarà disponibile" → HARD CONSTRAINT (divieto su tutti i turni
  di quel giorno).
Output:
model.Add(shifts[(2, 8, 0)] == 0)
model.Add(shifts[(2, 8, 1)] == 0)
model.Add(shifts[(2, 8, 2)] == 0)

--- Esempio 5 ---
Input: "Il lavoratore 1 di solito lavora bene di notte."
Ragionamento: "di solito lavora bene" → AMBIGUO. Non è chiaro se è una
  preferenza del lavoratore, un'osservazione, o un obbligo.
Output:
AMBIGUOUS: preferenza o obbligo non distinguibili dall'input

════════════════════════════════════════
ORA TRADUCI LA SEGUENTE RICHIESTA
════════════════════════════════════════
Input: {USER_INPUT}
Ragionamento:
Output:"""

    def translate_preference(self, user_input: str) -> str:
        """
        Invia la preferenza in linguaggio naturale a Gemini e restituisce il codice Python.
        """
        # Costruiamo il blocco finale da inviare come contenuto
        user_prompt = f"Input: {user_input}\nRagionamento:\n"

        # Nel nuovo SDK le configurazioni usano GenerateContentConfig
        # Passiamo il system_prompt direttamente come parametro di sistema
        config = types.GenerateContentConfig(
            temperature=0.1,
            top_p=0.9,
            system_instruction=self.system_prompt
        )

        try:
            # La chiamata ora passa dal client
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=config
            )

            raw_output = response.text.strip()

            # Estraiamo solo la parte di codice dopo "Output:" (ignorando il Chain-of-Thought)
            if "Output:" in raw_output:
                # Divide la stringa e prende tutto ciò che c'è dopo "Output:"
                final_code = raw_output.split("Output:")[1].strip()

                # Rimuove eventuali backtick di markdown (```python ... ```) che l'LLM potrebbe comunque inserire
                final_code = final_code.replace("```python", "").replace("```", "").strip()
                return final_code
            else:
                # Fallback se l'LLM non rispetta il formato per qualche motivo
                return raw_output

        except Exception as e:
            return f"AMBIGUOUS: Errore API Gemini - {str(e)}"