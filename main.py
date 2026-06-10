import os
import importlib.util
from src.system_builder import SystemBuilderAgent
from src.preferences.translator import PreferenceTranslator


def main():
    # Percorsi dei file
    draft_path = os.path.join("data", "model_draft.txt")
    preferences_path = os.path.join("data", "preferences.txt")
    model_output_path = os.path.join("src", "scheduler_model.py")

    print("==================================================")
    print(" AVVIO SMART SCHEDULER - AGENTIC FRAMEWORK")
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
    # Necessario perché scheduler_model.py è stato scritto su disco
    # a runtime: non esiste al momento dell'avvio del processo.
    print("\n[*] Caricamento dinamico di scheduler_model.py...")
    spec = importlib.util.spec_from_file_location("scheduler_model", model_output_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    SmartSchedulerModel = module.SmartSchedulerModel

    # ── FASE 2: Schedule Drafting ──────────────────────────
    print("\n[FASE 2] Costruzione e risoluzione del modello...")
    scheduler = SmartSchedulerModel(num_workers=13, num_days=31)
    scheduler.build_base_constraints()
    scheduler.apply_preferences()   # vincoli e pesi iniettati dalla Fase 1
    # scheduler.build_objective()   # TODO: Fase 2
    # scheduler.solve()             # TODO: Fase 2

    print("\n[+] Pipeline completata fino alla Fase 1.")


if __name__ == "__main__":
    main()