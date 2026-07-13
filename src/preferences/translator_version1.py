import ast

from dotenv import load_dotenv
import os
import ast
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
  - `w`: ID lavoratore (intero >= 0)
  - `d`: giorno dell'orizzonte di scheduling (da 0 a 30, dove 0 = 7 dicembre 2026)
  - `s`: turno -> 0 = Mattina (8-14), 1 = Pomeriggio (14-20), 2 = Notte (20-8)
- Dizionario dei pesi di soddisfazione (soft): `satisfaction_weights[(w, d, s)]`
  - Ogni peso è un intero. Un valore ALTO indica che assegnare quel turno è DESIDERATO.
  - Un valore BASSO (o negativo) indica che quel turno è INDESIDERATO.
  - La funzione obiettivo massimizza la somma pesata degli shift assegnati.
- Giorni di weekend nell'orizzonte: [5, 6, 12, 13, 19, 20, 26, 27, 30]
  (corrispondenti a sabati e domeniche dal 7 dicembre al 6 gennaio)

════════════════════════════════════════
REGOLE DI CLASSIFICAZIONE E GENERAZIONE
════════════════════════════════════════

Esistono TRE categorie. Classificale nell'ordine indicato.

CATEGORIA 1 — HARD CONSTRAINT PERSONALE (indisponibilità dichiarata dal lavoratore):
  Condizione: il lavoratore dichiara esplicitamente di non poter essere presente,
  per motivi personali, medici o contrattuali che esulano dalla legge già codificata.
  Parole chiave: "non può", "non sarà disponibile", "è impossibile per me",
  "ho un impegno", "non è disponibile il giorno X".
  Codice da generare:
    `model.add(shifts[(w, d, s)] == 0)`
  ATTENZIONE: usa questa categoria SOLO per divieti. Non esiste un hard constraint
  personale di tipo "obbligo" (es. "voglio lavorare il giorno X" è sempre una preferenza).

CATEGORIA 2 — SOFT CONSTRAINT (preferenza o desiderio):
  Condizione: il lavoratore esprime una preferenza, un desiderio, o una tolleranza.
  Non è un divieto assoluto: il solver può ignorarlo se necessario.
  Parole chiave: "preferisce", "vorrebbe evitare", "se possibile", "gradisce",
  "tollera", "non ama", "è disponibile per", "può lavorare durante".
  Codice da generare:
    - Turno desiderato:    `satisfaction_weights[(w, d, s)] = +10`
    - Turno indesiderato:  `satisfaction_weights[(w, d, s)] = -10`

CATEGORIA 3 — AMBIGUO:
  Condizione: non è possibile classificare con certezza nelle categorie 1 o 2.
  Risposta: `AMBIGUOUS: <descrizione breve del problema in meno di 10 parole>`

REGOLA CRITICA — cosa NON generare mai:
  - NON generare mai `model.add(shifts[(w, d, s)] == 1)`.
    Un lavoratore non può obbligare il sistema ad assegnargli un turno.
  - NON replicare vincoli già presenti nel model_draft.txt (limiti orari settimanali,
    riposo post-notte, ecc.). Quelli sono gestiti dal System Builder. Se la preferenza
    di un lavoratore coincide con un vincolo legale già codificato, ignorala silenziosamente.

════════════════════════════════════════
REGOLE DI OUTPUT
════════════════════════════════════════
- Rispondi ESCLUSIVAMENTE con codice Python puro.
- NON scrivere "Ragionamento:", "Output:", né alcuna etichetta.
- Se vuoi spiegare la classificazione, usa commenti Python (righe che iniziano con #).
- Non aggiungere NULLA prima o dopo il codice.
- Non usare blocchi markdown (```python ... ```).
- Se la richiesta è ambigua: AMBIGUOUS: <descrizione breve>

════════════════════════════════════════
ESEMPI (Few-Shot con Chain-of-Thought)
════════════════════════════════════════

--- Esempio 1 ---
Input: "Il lavoratore 3 non può assolutamente lavorare nel turno di Notte il giorno 15."
Output:
# Categoria 1: indisponibilità dichiarata. Giorno 15, turno Notte = s=2.
model.add(shifts[(3, 15, 2)] == 0)

--- Esempio 2 ---
Input: "Il lavoratore 0 preferisce i turni di mattina il giorno 2 e il giorno 3."
Output:
# Categoria 2: preferenza positiva su turni specifici.
satisfaction_weights[(0, 2, 0)] = 10
satisfaction_weights[(0, 3, 0)] = 10

--- Esempio 3 ---
Input: "Il lavoratore 5 non vuole lavorare nel weekend del primo fine settimana
  (giorni 5 e 6 dell'orizzonte)."
Output:
# Categoria 2: preferenza negativa. Non è un divieto assoluto.
satisfaction_weights[(5, 5, 0)] = -10
satisfaction_weights[(5, 5, 1)] = -10
satisfaction_weights[(5, 5, 2)] = -10
satisfaction_weights[(5, 6, 0)] = -10
satisfaction_weights[(5, 6, 1)] = -10
satisfaction_weights[(5, 6, 2)] = -10

--- Esempio 4 ---
Input: "Il lavoratore 2 ha un impegno personale e non sarà disponibile il giorno 8."
Output:
# Categoria 1: indisponibilità dichiarata su tutti i turni del giorno 8.
for s in range(3):
    model.add(shifts[(2, 8, s)] == 0)

--- Esempio 5 ---
Input: "Il lavoratore 1 di solito lavora bene di notte."
Output:
# Categoria 3: osservazione esterna, non è né una preferenza né un divieto.
AMBIGUOUS: preferenza o osservazione non distinguibili dall'input

--- Esempio 6 ---
Input: "Il lavoratore 1 preferisce i turni di mattina."
Output:
# Categoria 2: preferenza positiva estesa a tutti i giorni dell'orizzonte.
for d in range(31):
    satisfaction_weights[(1, d, 0)] = 10

--- Esempio 7 ---
Input: "Il lavoratore 2 è disponibile a lavorare durante i giorni festivi."
Output:
# Categoria 2: disponibilità espressa = preferenza positiva, non obbligo.
# Indici festivi nell'orizzonte: 1 (8 dic), 17 (24 dic), 18 (25 dic), 25 (1 gen), 30 (6 gen).
for s in range(3):
    satisfaction_weights[(2, 1, s)] = 10
    satisfaction_weights[(2, 17, s)] = 10
    satisfaction_weights[(2, 18, s)] = 10
    satisfaction_weights[(2, 25, s)] = 10
    satisfaction_weights[(2, 30, s)] = 10

════════════════════════════════════════
ORA TRADUCI LA SEGUENTE RICHIESTA
════════════════════════════════════════
Input: {USER_INPUT}
Output:"""


    def _extract_python_code(self, raw_output: str) -> str:
            markdown_pattern = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)
            match = markdown_pattern.search(raw_output)
            if match:
                candidate = match.group(1).strip()
                if self._is_valid_python(candidate):
                    return candidate


            PYTHON_STARTERS = (
                # keyword e strutture
                "for ", "if ", "else", "elif ", "while ", "def ", "class ",
                "import ", "from ", "return ", "with ", "try:", "except",
                "raise ", "pass", "break", "continue",
                # pattern OR-Tools specifici del progetto
                "model.", "shifts[", "satisfaction_weights[",
                # commenti
                "#",
            )

            clean_lines = []
            for line in raw_output.splitlines():
                stripped = line.strip()

                if not stripped:
                    clean_lines.append(line)
                    continue

                if line.startswith(("    ", "\t")):
                    clean_lines.append(line)
                    continue

                if any(stripped.startswith(p) for p in PYTHON_STARTERS):
                    clean_lines.append(line)
                    continue

                if self._is_valid_python(stripped):
                    clean_lines.append(line)
                    continue

                print(f"    [parser] Riga scartata (non-Python): {stripped[:60]}")

            candidate = "\n".join(clean_lines).strip()

            if not candidate:
                return "AMBIGUOUS: output LLM vuoto dopo il parsing"

            if self._is_valid_python(candidate):
                return candidate

            return f"AMBIGUOUS: codice estratto non è Python valido - {candidate[:100]}"


    def _is_valid_python(self, code: str) -> bool:
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _read_preferences_file(self, file_path: str) -> list[str]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"File preferenze non trovato: '{file_path}'"
            )

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        preferences = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                preferences.append(stripped)

        if not preferences:
            raise ValueError(
                f"Il file '{file_path}' non contiene preferenze valide."
            )

        return preferences

    def translate_preference(self, user_input: str) -> str:
        user_prompt = f"Input: {user_input}"

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
            if raw_output.startswith("AMBIGUOUS:"):
                return raw_output

            # Parsing a prova di bomba
            return self._extract_python_code(raw_output)

        except Exception as e:
            return f"AMBIGUOUS: Errore API Gemini - {str(e)}"

    def _inject_into_scheduler(
            self,
            scheduler_path: str,
            code_block: str,
    ) -> None:
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
        backup_path = scheduler_path + ".bak"
        shutil.copy2(scheduler_path, backup_path)
        print(f"  [backup] Creato '{backup_path}'")
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

    def process(
            self,
            preferences_path: str,
            scheduler_path: str,
    ) -> None:
        print(f"\n{'='*60}")
        print(f"  SmartScheduler — Fase 1: Preferences Definition")
        print(f"{'='*60}")
        print(f"  Preferenze:  {preferences_path}")
        print(f"  Scheduler:   {scheduler_path}\n")
        preferences = self._read_preferences_file(preferences_path)
        print(f"  {len(preferences)} preferenza/e trovata/e.\n")
        valid_snippets = []
        ambiguous_items = []

        for i, pref in enumerate(preferences, start=1):
            print(f"  [{i}/{len(preferences)}] {pref[:80]}{'...' if len(pref) > 80 else ''}")
            result = self.translate_preference(pref)

            if result.startswith("AMBIGUOUS:"):
                print(f"    [!]  {result}")
                ambiguous_items.append((pref, result))
            else:
                print(f"    [OK]  Codice generato ({len(result.splitlines())} righe)")
                snippet = f"# Preferenza: {pref}\n{result}"
                valid_snippets.append(snippet)

        if valid_snippets:
            full_code_block = "\n\n".join(valid_snippets)
            print(f"\n  Iniezione di {len(valid_snippets)} blocco/i in '{scheduler_path}'...")
            self._inject_into_scheduler(scheduler_path, full_code_block)
            print("  [OK]  Iniezione completata.")
        else:
            print("\n  Nessun codice valido da iniettare.")

        print(f"\n{'-'*60}")
        print(f"  REPORT FINALE")
        print(f"  Preferenze tradotte:  {len(valid_snippets)}")
        print(f"  Preferenze ambigue:   {len(ambiguous_items)}")

        if ambiguous_items:
            print("\n  [!]  Preferenze ambigue (richiesta revisione manuale):")
            for pref, reason in ambiguous_items:
                print(f"    - \"{pref[:60]}...\"")
                print(f"      -> {reason}")

        print(f"{'='*60}\n")


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