import os
import importlib.util
from ortools.sat.python import cp_model  # Aggiunto per decodificare lo stato del solver
from src.system_builder import SystemBuilderAgent
from src.preferences.translator import PreferenceTranslator
from src.validation.validation import HardConstraintVerifier # (o il nome del tuo file)
from src.validation.feedback import FeedbackPromptBuilder # (o il nome del tuo file)

def load_dynamic_model(model_output_path):
    """Funzione helper per ricaricare il modulo Python dinamicamente ad ogni iterazione."""
    spec = importlib.util.spec_from_file_location("scheduler_model", model_output_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SmartSchedulerModel

def main():
    draft_path = os.path.join("data", "model_draft.txt")
    preferences_path = os.path.join("data", "preferences.txt")
    model_output_path = os.path.join("src", "scheduler_model.py")

    print("==================================================")
    print("AVVIO SMART SCHEDULER - AGENTIC FRAMEWORK")
    print("==================================================")

    builder_agent = SystemBuilderAgent()
    translator = PreferenceTranslator()

    # Parametri del ciclo di Feedback
    MAX_RETRIES = 3
    attempt = 1
    is_schedule_valid = False
    feedback_prompt = None

    while attempt <= MAX_RETRIES and not is_schedule_valid:
        print(f"\n>>> INIZIO ITERAZIONE {attempt}/{MAX_RETRIES} <<<")

        # ── FASE 0: System Building / Revision ────────────────────
        if attempt == 1:
            print("[FASE 0] Generazione del modello base...")
            success = builder_agent.generate_model_file(draft_path, model_output_path)
        else:
            print("[FASE 0 - REVISION] Il Drafting Agent sta correggendo il modello...")
            success = builder_agent.generate_model_file(draft_path, model_output_path, feedback_prompt=feedback_prompt)

        if not success:
            print("\n[-] Generazione/Revisione fallita. Interruzione.")
            return

        # ── FASE 1: Preferences Definition ────────────────────
        print("[FASE 1] Iniezione delle preferenze...")
        translator.process(preferences_path, model_output_path)

        # ── FASE 2: Schedule Drafting ──────────────────────────
        print("[FASE 2] Costruzione e risoluzione del modello...")
        SmartSchedulerModel = load_dynamic_model(model_output_path)
        scheduler = SmartSchedulerModel(num_workers=13, num_days=31)

        scheduler.build_base_constraints()
        scheduler.apply_preferences()
        scheduler.build_objective()

        status, solver = scheduler.solve()

        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            print("[+] Soluzione matematica trovata. Passaggio alla validazione simbolica...")

            # ── ESTRAZIONE DIZIONARIO PER IL VERIFICATORE ──
            schedule_dict = {}
            for d in range(31):
                schedule_dict[d] = {}
                for s in range(3):
                    assigned_workers = []
                    for w in range(13):
                        if solver.Value(scheduler.shifts[(w, d, s)]) == 1:
                            assigned_workers.append(w)
                    schedule_dict[d][s] = assigned_workers

            # ── FASE 3: Schedule Verification ─────────────────
            print("[FASE 3] Esecuzione Hard Constraint Verifier...")
            verifier = HardConstraintVerifier(schedule_dict, num_workers=13, num_days=31)
            is_valid, validation_result = verifier.verify_all(min_workers=2, max_workers=3)

            if is_valid:
                print("\n[+] SUCCESSO! La turnazione rispetta tutti i vincoli legali.")
                is_schedule_valid = True
                # QUI IN FUTURO INSERIREMO LA FAIRNESS EVALUATION (FASE 4)
            else:
                print("\n[-] VALIDAZIONE FALLITA. Generazione prompt di feedback...")
                feedback_builder = FeedbackPromptBuilder()
                feedback_prompt = feedback_builder.build_revision_prompt(validation_result)
                print("--- PROMPT GENERATO PER L'AGENTE ---")
                print(feedback_prompt[:200] + "...\n[Prompt troncato per leggibilità]")

        else:
            print("[-] STATO SOLVER: INFEASIBLE/UNKNOWN. Generazione feedback per rilassare i vincoli...")
            # Se il solver fallisce, creiamo un feedback generico per l'agente
            feedback_prompt = "Il solver ha restituito INFEASIBLE. Controlla di non aver inserito vincoli matematicamente impossibili da soddisfare insieme. Rivedi il modello."

        attempt += 1

    if is_schedule_valid:
        print("\n==================================================")
        print("PIPELINE COMPLETATA CON SUCCESSO")
        print("==================================================")
        # Qui puoi stampare la turnazione finale se lo desideri
    else:
        print("\n==================================================")
        print("FALLIMENTO: Limite di tentativi raggiunto. Impossibile generare una schedule valida.")
        print("==================================================")

if __name__ == "__main__":
    main()