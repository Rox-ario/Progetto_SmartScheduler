from dotenv import load_dotenv
import os
import re
import shutil
import google.genai as genai
from google.genai import types

INJECTION_MARKER = "# <<< PREFERENCES_INJECTION_POINT >>>"


class PreferenceTranslator:
    def __init__(self, api_key=None):
        load_dotenv(dotenv_path=".env")
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API Key per Gemini mancante. "
                "Imposta la variabile d'ambiente GEMINI_API_KEY."
            )

        self.client = genai.Client(api_key=self.api_key)
        self.model_name = os.environ.get("MODEL_USED")

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
    `model.add(shifts[(w, d, s)] == 0)`
  - Per HARD CONSTRAINT (obbligo):
    `model.add(shifts[(w, d, s)] == 1)`
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
Ragionamento: "non può assolutamente" -> HARD CONSTRAINT (divieto). Giorno 15
  dell'orizzonte, turno Notte = s=2.
Output:
model.add(shifts[(3, 15, 2)] == 0)

--- Esempio 2 ---
Input: "Il lavoratore 0 preferisce i turni di mattina il giorno 2 e il giorno 3."
Ragionamento: "preferisce" -> SOFT CONSTRAINT (desiderato). Non è un obbligo.
  Aumento il peso di soddisfazione per quei turni.
Output:
satisfaction_weights[(0, 2, 0)] = 10
satisfaction_weights[(0, 3, 0)] = 10

--- Esempio 3 ---
Input: "Il lavoratore 5 non vuole lavorare nel weekend del primo fine settimana
  (giorni 5 e 6 dell'orizzonte)."
Ragionamento: "non vuole" -> SOFT CONSTRAINT (indesiderato). Non è un divieto
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
Ragionamento: "non sarà disponibile" -> HARD CONSTRAINT (divieto su tutti i turni
  di quel giorno).
Output:
for s in range(3):
    model.add(shifts[(2, 8, s)] == 0)

--- Esempio 5 ---
Input: "Il lavoratore 1 di solito lavora bene di notte."
Ragionamento: "di solito lavora bene" -> AMBIGUO. Non è chiaro se è una
  preferenza del lavoratore, un'osservazione, o un obbligo.
Output:
AMBIGUOUS: preferenza o obbligo non distinguibili dall'input

--- Esempio 6 ---
Input: "Il lavoratore 1 preferisce i turni di mattina."
Ragionamento: "preferisce" su tutti i giorni -> SOFT CONSTRAINT su d=0..30.
Output:
for d in range(31):
    satisfaction_weights[(1, d, 0)] = 10

--Esempio 7--
Input: "Il lavoratore 2 è disponibile/preferisce lavorare durante i giorni di vacanza"
Ragionamento: "preferisce" sui giorni -> SOFT CONSTRAINT su d=25,26,1,6.
Output:
for s in range(3):
    satisfaction_weights[(2, 25, s)] = 10
    satisfaction_weights[(2, 26, s)] = 10
    satisfaction_weights[(2, 1, s)] = 10
    satisfaction_weights[(2, 6, s)] = 10
════════════════════════════════════════
ORA TRADUCI LA SEGUENTE RICHIESTA
════════════════════════════════════════
Input: {USER_INPUT}
Ragionamento:
Output:"""

    # ─────────────────────────────────────────────
    # LETTURA FILE
    # ─────────────────────────────────────────────

    def _read_preferences_file(self, file_path: str) -> list[str]:
        """
        Legge il file delle preferenze e restituisce una lista di righe
        non vuote e non commentate (righe che iniziano con #).
        Ogni riga è trattata come una preferenza indipendente.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"File preferenze non trovato: '{file_path}'"
            )

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        preferences = []
        for line in lines:
            stripped = line.strip()
            # Salta righe vuote e commenti
            if stripped and not stripped.startswith("#"):
                preferences.append(stripped)

        if not preferences:
            raise ValueError(
                f"Il file '{file_path}' non contiene preferenze valide."
            )

        return preferences

    # ─────────────────────────────────────────────
    # TRADUZIONE SINGOLA PREFERENZA
    # ─────────────────────────────────────────────

    def translate_preference(self, user_input: str) -> str:
        """
        Invia una singola preferenza in linguaggio naturale a Gemini
        e restituisce il codice Python corrispondente (o una stringa AMBIGUOUS:).
        """
        user_prompt = f"Input: {user_input}\nRagionamento:\nOutput:"

        config = types.GenerateContentConfig(
            temperature=0.1,
            top_p=0.9,
            system_instruction=self.system_prompt,
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=config,
            )
            raw_output = response.text.strip()

            # Estrae solo la parte dopo "Output:" ignorando il chain-of-thought
            if "Output:" in raw_output:
                final_code = raw_output.split("Output:")[-1].strip()
            else:
                final_code = raw_output

            # Rimuove eventuali backtick markdown residui
            final_code = re.sub(r"```(?:python)?", "", final_code).replace("```", "").strip()
            return final_code

        except Exception as e:
            return f"AMBIGUOUS: Errore API Gemini - {str(e)}"

    # ─────────────────────────────────────────────
    # INIEZIONE NEL FILE SCHEDULER
    # ─────────────────────────────────────────────

    def _inject_into_scheduler(
            self,
            scheduler_path: str,
            code_block: str,
    ) -> None:
        """
        Legge scheduler_model.py, individua il marcatore di iniezione,
        e inserisce il codice generato subito sotto di esso.
        Crea un backup (.bak) prima di modificare il file.
        """
        if not os.path.exists(scheduler_path):
            raise FileNotFoundError(
                f"File scheduler non trovato: '{scheduler_path}'"
            )

        with open(scheduler_path, "r", encoding="utf-8") as f:
            original_content = f.read()

        if INJECTION_MARKER not in original_content:
            raise ValueError(
                f"Marcatore di iniezione '{INJECTION_MARKER}' non trovato "
                f"in '{scheduler_path}'.\n"
                "Aggiungi manualmente quella riga nel punto corretto del file."
            )

        # Backup prima di toccare qualsiasi cosa
        backup_path = scheduler_path + ".bak"
        shutil.copy2(scheduler_path, backup_path)
        print(f"  [backup] Creato '{backup_path}'")

        # Il marcatore si trova dentro apply_preferences() → 8 spazi fissi.
        # Questo è il contratto con il LLM: NON cambiare questo valore
        # se non cambi anche la posizione del marcatore nello scheduler.
        INDENT = " " * 8
        indented_code = "\n".join(
            (INDENT + line if line.strip() else line)
            for line in code_block.splitlines()
        )

        new_content = original_content.replace(
            INJECTION_MARKER,
            INJECTION_MARKER + "\n" + indented_code,
            )

        with open(scheduler_path, "w", encoding="utf-8") as f:
            f.write(new_content)

    # ─────────────────────────────────────────────
    # ENTRY POINT PRINCIPALE
    # ─────────────────────────────────────────────

    def process(
            self,
            preferences_path: str,
            scheduler_path: str,
    ) -> None:
        """
        Pipeline completa:
          1. Legge le preferenze dal file di testo
          2. Traduce ciascuna preferenza con Gemini
          3. Inietta il codice valido nel file scheduler
          4. Stampa un report finale

        Le righe AMBIGUOUS non vengono iniettate ma vengono segnalate.
        """
        print(f"\n{'='*60}")
        print(f"  SmartScheduler — Fase 1: Preferences Definition")
        print(f"{'='*60}")
        print(f"  Preferenze:  {preferences_path}")
        print(f"  Scheduler:   {scheduler_path}\n")

        # Step 1 — lettura
        preferences = self._read_preferences_file(preferences_path)
        print(f"  {len(preferences)} preferenza/e trovata/e.\n")

        # Step 2 — traduzione
        valid_snippets = []
        ambiguous_items = []

        for i, pref in enumerate(preferences, start=1):
            print(f"  [{i}/{len(preferences)}] {pref[:80]}{'...' if len(pref) > 80 else ''}")
            result = self.translate_preference(pref)

            if result.startswith("AMBIGUOUS:"):
                print(f"    ⚠  {result}")
                ambiguous_items.append((pref, result))
            else:
                print(f"    ✓  Codice generato ({len(result.splitlines())} righe)")
                # Aggiunge un commento di provenienza per leggibilità nel file finale
                snippet = f"# Preferenza: {pref}\n{result}"
                valid_snippets.append(snippet)

        # Step 3 — iniezione
        if valid_snippets:
            full_code_block = "\n\n".join(valid_snippets)
            print(f"\n  Iniezione di {len(valid_snippets)} blocco/i in '{scheduler_path}'...")
            self._inject_into_scheduler(scheduler_path, full_code_block)
            print("  ✓  Iniezione completata.")
        else:
            print("\n  Nessun codice valido da iniettare.")

        # Step 4 — report
        print(f"\n{'─'*60}")
        print(f"  REPORT FINALE")
        print(f"  Preferenze tradotte:  {len(valid_snippets)}")
        print(f"  Preferenze ambigue:   {len(ambiguous_items)}")

        if ambiguous_items:
            print("\n  ⚠  Preferenze ambigue (richiesta revisione manuale):")
            for pref, reason in ambiguous_items:
                print(f"    - \"{pref[:60]}...\"")
                print(f"      → {reason}")

        print(f"{'='*60}\n")


# ─────────────────────────────────────────────
# USO DA RIGA DI COMANDO
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SmartScheduler — Fase 1: traduce preferenze e le inietta nello scheduler."
    )
    parser.add_argument(
        "preferences_file",
        help="Path al file .txt con le preferenze (una per riga).",
    )
    parser.add_argument(
        "scheduler_file",
        help="Path al file scheduler_model.py da modificare.",
    )
    args = parser.parse_args()

    translator = PreferenceTranslator()
    translator.process(args.preferences_file, args.scheduler_file)