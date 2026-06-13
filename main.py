import os
import importlib.util
from ortools.sat.python import cp_model  # Aggiunto per decodificare lo stato del solver
from src.system_builder import SystemBuilderAgent
from src.preferences.translator import PreferenceTranslator


def main():
    # Percorsi dei file
    draft_path = os.path.join("data", "model_draft.txt")
    preferences_path = os.path.join("data", "preferences.txt")
    model_output_path = os.path.join("src", "scheduler_model.py")

    print("==================================================")
    print("AVVIO SMART SCHEDULER - AGENTIC FRAMEWORK")
    print("==================================================")

    # ── FASE 0: System Building ────────────────────────────
    print("\n[FASE 0] Generazione del modello base...")
    builder_agent = SystemBuilderAgent()
    success = builder_agent.generate_model_file(draft_path, model_output_path)

    if not success:
        print("\n[-] Generazione del modello fallita. Il framework si interrompe.")
        return

    print("[+] scheduler_model.py generato correttamente.")

    # ── FASE 1: Preferences Definition ────────────────────
    print("\n[FASE 1] Traduzione e iniezione delle preferenze...")
    translator = PreferenceTranslator()
    translator.process(preferences_path, model_output_path)

    # ── IMPORT DINAMICO ────────────────────────────────────
    print("\n[*] Caricamento dinamico di scheduler_model.py...")
    spec = importlib.util.spec_from_file_location("scheduler_model", model_output_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    SmartSchedulerModel = module.SmartSchedulerModel

    # ── FASE 2: Schedule Drafting ──────────────────────────
    print("\n[FASE 2] Costruzione e risoluzione del modello...")
    scheduler = SmartSchedulerModel(num_workers=13, num_days=31)

    print("[*] Costruzione dei vincoli base (Hard Constraints)...")
    scheduler.build_base_constraints()

    print("[*] Applicazione delle preferenze (Soft Constraints)...")
    scheduler.apply_preferences()

    print("[*] Costruzione della funzione obiettivo...")
    scheduler.build_objective()

    print("[*] Avvio del Solver OR-Tools (Ricerca della soluzione)...")
    status, solver = scheduler.solve()

    # ── FASE 3: Estrazione e Output ─────────────────────────
    print("\n==================================================")
    print("RISULTATI DELLA SCHEDULAZIONE")
    print("==================================================")

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        esito = "OTTIMO (Massima soddisfazione possibile)" if status == cp_model.OPTIMAL else "FEASIBLE (Soluzione valida trovata)"
        print(f"[+] Stato: {esito}")
        print(f"[+] Punteggio funzione obiettivo: {solver.ObjectiveValue()}\n")

        # Dizionario per tradurre gli indici dei turni in testo leggibile
        shift_names = {0: "Mattina  (08-14)", 1: "Pomeriggio(14-20)", 2: "Notte    (20-08)"}

        # Estraiamo i turni assegnati giorno per giorno
        for d in range(31):
            print(f"--- GIORNO {d} ---")
            for s in range(3):
                assigned_workers = []
                for w in range(13):
                    # Se il solver ha impostato questa variabile booleana a 1 (True), il lavoratore fa il turno
                    if solver.Value(scheduler.shifts[(w, d, s)]) == 1:
                        assigned_workers.append(f"Lavoratore {w}")

                # Formattiamo l'output
                if assigned_workers:
                    print(f"  {shift_names[s]}: {', '.join(assigned_workers)}")
                else:
                    print(f"  {shift_names[s]}: SCOPERTO")
            print() # Riga vuota di separazione

    elif status == cp_model.INFEASIBLE:
        print("[-] STATO: INFEASIBLE")
        print("[-] Impossibile trovare una turnazione valida. I vincoli rigidi (Hard Constraints) sono troppo restrittivi o in conflitto tra loro.")
    else:
        print("[-] STATO: UNKNOWN / MODEL_INVALID")
        print("[-] Il solver non è riuscito a concludere la ricerca.")

if __name__ == "__main__":
    main()