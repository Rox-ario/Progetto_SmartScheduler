import os
import google.genai as genai
from dotenv import load_dotenv
import time
from google.genai import types


class SystemBuilderAgent:
    def __init__(self, api_key=None):
        load_dotenv(dotenv_path=".env")

        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("API Key per Gemini mancante. Imposta GEMINI_API_KEY.")

        self.client = genai.Client(api_key=self.api_key)
        self.model_name = os.environ.get("MODEL_USED")

        self.system_prompt = """Sei un Senior Software Engineer specializzato in Ricerca Operativa e Google OR-Tools (CP-SAT).
Il tuo compito è leggere un documento di specifiche (Model Draft) e generare un file Python completo, funzionante e privo di errori.

════════════════════════════════════════
ARCHITETTURA OBBLIGATORIA DELLA CLASSE
════════════════════════════════════════

Devi generare una classe `SmartSchedulerModel` con ESATTAMENTE questi metodi, nell'ordine indicato:

━━━ 1. __init__(self, num_workers, num_days) ━━━
- Inizializza `self.model = cp_model.CpModel()`
- Crea `self.shifts = {}` con variabili booleane `new_bool_var` per ogni (w, d, s)
- Crea `self.satisfaction_weights = {}` inizializzato a 0 per ogni (w, d, s)
- Definisce `self.shift_durations` e `self.shift_weights` come dizionari
- Per la "rolling 7-day window" usa: `for d in range(self.num_days - 6):`

━━━ 2. build_base_constraints(self) ━━━
- Implementa TUTTI gli Hard Constraints definiti nel draft
- Usa `self.model`, `self.shifts` per tutti i vincoli
- Usa `self.model.add(...)` per vincoli lineari
- Usa `self.model.add_implication(...)` per implicazioni booleane

━━━ 3. apply_preferences(self) ━━━
QUESTO METODO È OBBLIGATORIO E DEVE ESSERE ESATTAMENTE COSÌ:

    def apply_preferences(self):
        \"\"\"
        Metodo dedicato all'iniezione delle preferenze generate dalla Fase 1.
        Gli alias locali espongono gli attributi della classe con i nomi
        che il codice generato da Gemini si aspetta.
        NON modificare i nomi degli alias: sono il contratto con il LLM.
        \"\"\"
        model = self.model
        shifts = self.shifts
        satisfaction_weights = self.satisfaction_weights

        # <<< PREFERENCES_INJECTION_POINT >>>

━━━ 4. build_objective(self) ━━━
- Costruisce la funzione obiettivo che massimizza la somma pesata degli shift assegnati
- Usa `self.satisfaction_weights[(w, d, s)]` come coefficienti
- Usa `self.model.maximize(cp_model.LinearExpr.weighted_sum(...))`
- NON invocare il solver qui

━━━ 5. solve(self) ━━━
- Crea `solver = cp_model.CpSolver()`
- Invoca `status = solver.solve(self.model)`
- Restituisce una tupla `(status, solver)` senza stampare nulla
- Aggiungi il limite di tempo di 120 secondi: `solver.parameters.max_time_in_seconds = 120.0`

════════════════════════════════════════
REGOLE CRITICHE
════════════════════════════════════════

- Il marcatore `# <<< PREFERENCES_INJECTION_POINT >>>` deve apparire nel corpo
  di `apply_preferences` ESATTAMENTE come scritto sopra, con 8 spazi di indentazione.
  Non aggiungere nulla dopo il marcatore all'interno del metodo.

- `apply_preferences` NON deve contenere nessun vincolo: è un metodo vuoto
  (a parte gli alias e il marcatore) che verrà popolato esternamente.

- Usa `from ortools.sat.python import cp_model` come unico import.

- CRITICO - Negazione di BoolVar: usa SEMPRE `.Not()` (N maiuscola).
  NON usare mai `.not()` (n minuscola): `not` è una keyword Python riservata
  e causa SyntaxError immediato. Esempio corretto:
  `self.model.add_implication(var_a, var_b.Not())`

════════════════════════════════════════
REGOLE DI OUTPUT
════════════════════════════════════════

- Restituisci ESCLUSIVAMENTE codice Python puro.
- Nessun blocco markdown (```python ... ```).
- Nessuna introduzione, nessun commento finale, nessuna spiegazione.
- Se l'indentazione è sbagliata o il marcatore manca, il sistema andrà in crash."""

    def generate_model_file(self, draft_filepath: str, output_filepath: str, feedback_prompt: str = None):
        """
        Legge il draft, invia a Gemini con Exponential Backoff, e salva il codice.
        Se feedback_prompt è fornito, agisce in modalità "Revisione" per correggere gli errori.
        """
        print(f"[*] Lettura del draft da: {draft_filepath}...")
        try:
            with open(draft_filepath, 'r', encoding='utf-8') as file:
                draft_content = file.read()
        except FileNotFoundError:
            print(f"Errore: Impossibile trovare il file {draft_filepath}")
            return False

        # --- GESTIONE DEL FEEDBACK LOOP ---
        if feedback_prompt:
            print("[*] Modalità REVISIONE: Integrazione del feedback nel prompt...")
            user_message = (
                f"{feedback_prompt}\n"
                f"----------------------------------------\n"
                f"Per tuo riferimento, ecco il Model Draft originale per non perdere i vincoli base:\n\n{draft_content}"
            )
        else:
            print("[*] Modalità GENERAZIONE: Creazione modello base...")
            user_message = f"Ecco il Model Draft da tradurre in codice:\n\n{draft_content}"

        print("[*] Contattando l'Agente LLM (System Builder) per generare il codice OR-Tools...")
        config = types.GenerateContentConfig(
            temperature=0.1,  # Teniamo la temperatura bassa per codice deterministico
            top_p=0.9,
            system_instruction=self.system_prompt
        )

        # --- GESTIONE EXPONENTIAL BACKOFF CON LIMITE MASSIMO ---
        max_retries = 5        # Numero massimo di tentativi prima di arrendersi
        max_allowed_wait = 300 # Attesa massima cumulativa in secondi (5 minuti)
        current_wait = 2       # Tempo di attesa base per il primo fallimento
        total_waited = 0       # Tempo totale già aspettato
        response = None

        for retry_attempt in range(max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=user_message,
                    config=config
                )
                break  # Se la chiamata va a buon fine, rompiamo il ciclo

            except Exception as e:
                error_msg = str(e)
                is_503 = "503" in error_msg
                is_429 = "429" in error_msg
                if is_503 or is_429:
                    code = "503 (Server Overload)" if is_503 else "429 (Rate Limit / Quota Esaurita)"
                    if retry_attempt >= max_retries or total_waited >= max_allowed_wait:
                        print(f"\n[-] ERRORE CRITICO [{code}]: fallito dopo {retry_attempt} tentativi ({total_waited}s di attesa totale).")
                        if is_429:
                            print("    → Possibile causa: quota giornaliera/al-minuto esaurita per questa API Key.")
                            print("    → Controlla: https://aistudio.google.com/app/apikey")
                        print(f"    → Dettaglio errore completo: {error_msg}")
                        return False

                    print(f"[-] {code} - tentativo {retry_attempt + 1}/{max_retries}. Attendo {current_wait}s...")
                    time.sleep(current_wait)
                    total_waited += current_wait
                    current_wait = min(current_wait * 2, max_allowed_wait - total_waited)
                else:
                    print(f"[-] Errore fatale durante la generazione:\n    {error_msg}")
                    return False
        # -------------------------------------------------------

        # Pulizia dell'output
        python_code = response.text.strip()
        if python_code.startswith("```python"):
            python_code = python_code[9:]
        if python_code.startswith("```"):
            python_code = python_code[3:]
        if python_code.endswith("```"):
            python_code = python_code[:-3]
        python_code = python_code.strip()

        # Verifica che il marcatore sia presente prima di salvare
        marker = "# <<< PREFERENCES_INJECTION_POINT >>>"
        if marker not in python_code:
            print(
                "[-] ATTENZIONE: Il codice generato non contiene il marcatore "
                f"'{marker}'.\n"
                "    La Fase 1 (preferenze) non potrà essere iniettata.\n"
                "    Il file NON è stato salvato. Riprova."
            )
            return False

        with open(output_filepath, 'w', encoding='utf-8') as out_file:
            out_file.write(python_code)

        print(f"[+] Codice generato con successo e salvato in: {output_filepath}")
        return True