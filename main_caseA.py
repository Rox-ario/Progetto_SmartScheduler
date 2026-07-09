import os
import importlib.util
from ortools.sat.python import cp_model # Aggiunto per decodificare lo stato del solver
from src.system_builder import SystemBuilderAgent
from src.preferences.translator import PreferenceTranslator
from src.validation.feedback import FeedbackPromptBuilder, RefinementPromptBuilder
from src.validation.validation_caseA import HardConstraintVerifier, FairnessEvaluationAgent

def print_schedule(schedule_dict, num_days=31):
    """
    Formatta e stampa a schermo il calendario generato dal solver.
    """
    print("\n" + "="*50)
    print("CALENDARIO SCHEDULING GENERATO".center(50))
    print("="*50)

    shift_names = ["Mattina (08-14)", "Pomeriggio (14-20)", "Notte (20-08)  "]

    for d in range(num_days):
        print(f"\n--- Giorno {d + 1} ---")
        for s in range(3):
            workers = schedule_dict[d][s]
            # Formattazione per rendere la lista dei lavoratori ben leggibile
            workers_str = ", ".join([f"W{w}" for w in workers]) if workers else "NESSUNO"
            print(f"  {shift_names[s]} : {workers_str}")
    print("\n" + "="*50 + "\n")

def load_dynamic_model(model_output_path):
    """Funzione helper per ricaricare il modulo Python dinamicamente ad ogni iterazione."""
    spec = importlib.util.spec_from_file_location("scheduler_model", model_output_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SmartSchedulerModel

def extract_schedule(solver, scheduler, num_workers, num_days=31):
    """Estrae il dizionario schedule dal solver risolto."""
    schedule_dict = {}
    for d in range(num_days):
        schedule_dict[d] = {}
        for s in range(3):
            assigned_workers = []
            for w in range(num_workers):
                if solver.Value(scheduler.shifts[(w, d, s)]) == 1:
                    assigned_workers.append(w)
            schedule_dict[d][s] = assigned_workers
    return schedule_dict

def extract_preferences(scheduler, num_workers):
    """Estrae le preferenze positive/negative dalla matrice satisfaction_weights."""
    extracted_preferences = {w: {'positive': [], 'negative': []} for w in range(num_workers)}
    for (w, d, s), weight in scheduler.satisfaction_weights.items():
        if weight > 0:
            extracted_preferences[w]['positive'].append((d, s))
        elif weight < 0:
            extracted_preferences[w]['negative'].append((d, s))
    return extracted_preferences

def build_and_solve(model_output_path, num_workers, fairness_lower_bounds=None, use_maxmin_objective=False):
    """
    Carica dinamicamente il modello, inietta le preferenze, costruisce e risolve.
    Restituisce (status, solver, scheduler) oppure (None, None, None) in caso di errore.

    fairness_lower_bounds: dict opzionale {worker_id: min_score} che forza
    programmaticamente un lower-bound sul satisfaction_score di ogni worker
    indicato. Viene applicato DOPO apply_preferences() così i pesi sono definitivi.
    Questo bypassa l'LLM per la parte numerica critica del refinement.

    use_maxmin_objective: se True, sostituisce l'obiettivo dell'LLM con un
    obiettivo max-min: massimizza il minimo satisfaction_score tra tutti i
    lavoratori. Questo spinge il solver a migliorare il worker più svantaggiato
    il più possibile, non solo di +1.
    """
    try:
        SmartSchedulerModel = load_dynamic_model(model_output_path)
        scheduler = SmartSchedulerModel(num_workers=num_workers, num_days=31)

        scheduler.build_base_constraints()
        scheduler.apply_preferences()

        # ── Iniezione programmatica dei lower-bound di fairness ──
        if fairness_lower_bounds:
            for w, min_score in fairness_lower_bounds.items():
                # Usiamo LinearExpr.weighted_sum invece di sum() per garantire
                # che score_w sia sempre un LinearExpr OR-Tools (mai un int Python),
                # anche quando tutti i pesi sono 0.
                score_w = cp_model.LinearExpr.weighted_sum(
                    [scheduler.shifts[(w, d, s)] for d in range(31) for s in range(3)],
                    [scheduler.satisfaction_weights.get((w, d, s), 0) for d in range(31) for s in range(3)]
                )
                scheduler.model.add(score_w >= min_score)
                print(f"    [fairness] Worker {w}: score >= {min_score} (iniettato programmaticamente)")

        if use_maxmin_objective:
            # ── Obiettivo Max-Min Fairness ──
            # Crea una variabile z = min(score_w per ogni w), poi massimizza z.
            # Così il solver porta il lavoratore più svantaggiato il più in alto
            # possibile, invece di limitarsi a soddisfare score >= min+1.
            z = scheduler.model.new_int_var(-10000, 10000, 'min_satisfaction_score')
            for w in range(num_workers):
                score_w = cp_model.LinearExpr.weighted_sum(
                    [scheduler.shifts[(w, d, s)] for d in range(31) for s in range(3)],
                    [scheduler.satisfaction_weights.get((w, d, s), 0) for d in range(31) for s in range(3)]
                )
                scheduler.model.add(z <= score_w)  # z è sempre <= score di ogni worker
            scheduler.model.maximize(z)             # massimizza il minimo
            print("    [objective] Obiettivo max-min fairness attivo.")
        else:
            scheduler.build_objective()

        status, solver = scheduler.solve()
        return status, solver, scheduler
    except Exception as e:
        print(f"[-] Errore durante la costruzione/risoluzione del modello: {e}")
        return None, None, None


def main():
    draft_path = os.path.join("data", "model_draft_caseA.txt")
    preferences_path = os.path.join("data", "preferences.txt")
    model_output_path = os.path.join("src", "scheduler_model.py")

    print("==================================================")
    print("AVVIO SMART SCHEDULER - AGENTIC FRAMEWORK")
    print("==================================================")

    builder_agent = SystemBuilderAgent()
    translator = PreferenceTranslator()

    # Parametri del ciclo di Feedback (Fasi 0-3)
    MAX_RETRIES = 10
    attempt = 1
    is_schedule_valid = False
    feedback_prompt = None
    num_workers=13
    schedule_dict = {}
    scheduler = None

    # ═══════════════════════════════════════════════════════
    # CICLO FASI 0-3: Generazione e Validazione Hard Constraints
    # ═══════════════════════════════════════════════════════
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
        status, solver, scheduler = build_and_solve(model_output_path, num_workers)  # Fase 0-3: senza lower-bound

        if status is None:
            feedback_prompt = "Il modello ha generato un errore Python durante l'esecuzione. Rivedi la sintassi e la struttura del codice."
            attempt += 1
            continue

        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            print("[+] Soluzione matematica trovata. Passaggio alla validazione simbolica...")

            # ── ESTRAZIONE DIZIONARIO PER IL VERIFICATORE ──
            schedule_dict = extract_schedule(solver, scheduler, num_workers)
            print_schedule(schedule_dict, num_days=31)

            # ── FASE 3: Schedule Verification ─────────────────
            print("[FASE 3] Esecuzione Hard Constraint Verifier...")
            verifier = HardConstraintVerifier(schedule_dict, num_workers=num_workers, num_days=31)
            is_valid, validation_result = verifier.verify_all(2)

            if is_valid:
                print("\n[+] SUCCESSO! La turnazione rispetta tutti i vincoli legali.")
                is_schedule_valid = True
            else:
                print("\n[-] VALIDAZIONE FALLITA. Generazione prompt di feedback...")
                feedback_builder = FeedbackPromptBuilder()
                feedback_prompt = feedback_builder.build_revision_prompt(validation_result)
                print("--- PROMPT GENERATO PER L'AGENTE ---")
                print(feedback_prompt)

        else:
            print("[-] STATO SOLVER: INFEASIBLE/UNKNOWN. Generazione feedback per rilassare i vincoli...")
            feedback_prompt = "Il solver ha restituito INFEASIBLE. Controlla di non aver inserito vincoli matematicamente impossibili da soddisfare insieme. Rivedi il modello."

        attempt += 1

    if not is_schedule_valid:
        print("\n==================================================")
        print("FALLIMENTO: Limite di tentativi raggiunto. Impossibile generare una schedule valida.")
        print("==================================================")
        return

    # ═══════════════════════════════════════════════════════
    # FASE 4: Schedule Refinement (Ciclo Iterativo di Fairness)
    # ═══════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("FASE 4: SCHEDULE REFINEMENT - CICLO DI FAIRNESS".center(60))
    print("="*60)

    MAX_REFINEMENT_ITERATIONS = 10
    refinement_builder = RefinementPromptBuilder()

    # Dizionario che accumula i lower-bound confermati iterazione per iterazione.
    # La chiave è il worker_id, il valore è il min_score che il solver DEVE superare.
    # Viene passato a build_and_solve in modo da iniettarlo direttamente nel modello
    # OR-Tools, senza delegare questo compito all'LLM (che potrebbe ignorarlo).
    fairness_lower_bounds = {}

    # Riferimento allo scheduler dell'ultima soluzione ACCETTATA come valida.
    # 'scheduler' viene sovrascritto ad ogni build_and_solve (anche se la soluzione
    # viene poi rifiutata o restituisce None), quindi non è sicuro usarlo nel report
    # finale. 'best_scheduler' viene aggiornato solo quando il miglioramento è
    # confermato, garantendo coerenza con 'schedule_dict'.
    best_scheduler = scheduler

    for refinement_iter in range(1, MAX_REFINEMENT_ITERATIONS + 1):
        print(f"\n{'─'*60}")
        print(f"  ITERAZIONE REFINEMENT {refinement_iter}/{MAX_REFINEMENT_ITERATIONS}")
        print(f"{'─'*60}")

        # ── 4.1: Valutazione Fairness (Agente Simbolico) ──
        extracted_preferences = extract_preferences(scheduler, num_workers)
        fairness_agent = FairnessEvaluationAgent(
            schedule_dict,
            num_workers=num_workers,
            num_days=31,
            preferences=extracted_preferences,
            satisfaction_weights=scheduler.satisfaction_weights
        )
        fairness_results = fairness_agent.evaluate_fairness()

        most_disadvantaged = fairness_results['most_disadvantaged_worker_id']
        satisfaction_scores = fairness_results['satisfaction_scores']
        current_min_score = fairness_results['min_satisfaction_score']

        print("\n--- RISULTATI FAIRNESS ---")
        print(f"Media turni disagiati:       {fairness_results['mean_disadvantaged_shifts']}")
        print(f"Deviazione Standard:         {fairness_results['standard_deviation']}")
        print(f"Lavoratore più svantaggiato: Worker {most_disadvantaged}")
        print(f"Suo satisfaction_score:       {current_min_score}")
        print(f"Satisfaction scores:         {satisfaction_scores}")
        print("--------------------------")

        # ── 4.2: Lettura del codice corrente del modello ──
        with open(model_output_path, 'r', encoding='utf-8') as f:
            current_code = f.read()

        # ── 4.3: Costruzione del prompt di refinement ──
        refinement_prompt = refinement_builder.build_refinement_prompt(
            most_disadvantaged_worker_id=most_disadvantaged,
            current_min_score=current_min_score,
            satisfaction_scores=satisfaction_scores,
            current_code=current_code
        )

        print("[FASE 4] Invio del prompt di refinement al Drafting Agent...")

        # ── 4.4: Il Drafting Agent (LLM) rigenera il codice ──
        success = builder_agent.generate_model_file(
            draft_path,
            model_output_path,
            feedback_prompt=refinement_prompt
        )

        if not success:
            print("[-] Il Drafting Agent non è riuscito a generare il codice raffinato.")
            print("[*] Si mantiene la schedulazione dell'iterazione precedente.")
            break

        # ── 4.5: Re-iniezione delle preferenze e risoluzione ──
        print("[FASE 4] Re-iniezione preferenze e risoluzione del modello raffinato...")
        translator.process(preferences_path, model_output_path)

        status, solver, scheduler = build_and_solve(
            model_output_path, num_workers,
            fairness_lower_bounds=fairness_lower_bounds,
            use_maxmin_objective=True
        )

        if status is None:
            print("[-] Errore nell'esecuzione del modello raffinato.")
            print("[*] Si mantiene la schedulazione dell'iterazione precedente.")
            break

        # ── 4.6: Controllo risultato solver ──
        if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            print("\n[*] SOLVER INFEASIBLE: Non è possibile migliorare ulteriormente la fairness")
            print("    senza violare i vincoli rigidi.")
            print("    Il ciclo di refinement si conclude. La schedulazione precedente è il risultato finale.")
            break

        print("[+] Nuova soluzione trovata. Verifica hard constraints...")

        # ── 4.7: Re-verifica hard constraints ──
        new_schedule_dict = extract_schedule(solver, scheduler, num_workers)
        verifier = HardConstraintVerifier(new_schedule_dict, num_workers=num_workers, num_days=31)
        is_valid, validation_result = verifier.verify_all(2)

        if not is_valid:
            print("[-] La nuova schedulazione viola i vincoli hard! Interruzione del refinement.")
            print(f"    Violazioni: {validation_result}")
            print("[*] Si mantiene la schedulazione dell'iterazione precedente.")
            break

        # ── 4.8: Verifica che la fairness sia effettivamente migliorata ──
        new_extracted_prefs = extract_preferences(scheduler, num_workers)
        new_fairness_agent = FairnessEvaluationAgent(
            new_schedule_dict,
            num_workers=num_workers,
            num_days=31,
            preferences=new_extracted_prefs,
            satisfaction_weights=scheduler.satisfaction_weights
        )
        new_fairness_results = new_fairness_agent.evaluate_fairness()
        new_min_score = new_fairness_results['min_satisfaction_score']

        print(f"\n[*] Score minimo precedente: {current_min_score}")
        print(f"[*] Score minimo attuale:    {new_min_score}")

        if new_min_score > current_min_score:
            print(f"[+] MIGLIORAMENTO CONFERMATO! (+{new_min_score - current_min_score})")
            # Aggiorniamo la schedulazione corrente e il riferimento al best scheduler
            schedule_dict = new_schedule_dict
            best_scheduler = scheduler  # ← aggiorna solo qui, su soluzione accettata
            print_schedule(schedule_dict, num_days=31)

            # ── Aggiorniamo i lower-bound per la prossima iterazione ──
            # Per ogni worker forziamo score >= score_attuale (non peggiorare)
            # e per il worker più svantaggiato score >= new_min_score + 1 (migliorare)
            for w, score in new_fairness_results['satisfaction_scores'].items():
                # Il lower-bound è il massimo tra quello già noto e lo score attuale
                fairness_lower_bounds[w] = max(fairness_lower_bounds.get(w, score), score)
            # Il worker più svantaggiato deve migliorare ulteriormente la prossima volta
            fairness_lower_bounds[most_disadvantaged] = new_min_score + 1
            print(f"    [fairness] Lower-bounds aggiornati: {fairness_lower_bounds}")
        else:
            print("[*] Nessun miglioramento effettivo. Il ciclo di refinement si conclude.")
            break
    else:
        print(f"\n[*] Raggiunto il limite massimo di {MAX_REFINEMENT_ITERATIONS} iterazioni di refinement.")

    # ═══════════════════════════════════════════════════════
    # REPORT FINALE
    # ═══════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("PIPELINE COMPLETATA CON SUCCESSO".center(60))
    print("="*60)
    print("\nSchedulazione finale:")
    print_schedule(schedule_dict, num_days=31)

    # Fairness report finale — usa best_scheduler (sempre coerente con schedule_dict)
    final_prefs = extract_preferences(best_scheduler, num_workers)
    final_fairness = FairnessEvaluationAgent(
        schedule_dict,
        num_workers=num_workers,
        num_days=31,
        preferences=final_prefs,
        satisfaction_weights=best_scheduler.satisfaction_weights
    )
    final_results = final_fairness.evaluate_fairness()

    print("--- REPORT FAIRNESS FINALE ---")
    print(f"Media turni disagiati:       {final_results['mean_disadvantaged_shifts']}")
    print(f"Deviazione Standard:         {final_results['standard_deviation']}")
    print(f"Lavoratore più svantaggiato: Worker {final_results['most_disadvantaged_worker_id']}")
    print(f"Min satisfaction_score:       {final_results['min_satisfaction_score']}")
    print(f"Tutti gli score:             {final_results['satisfaction_scores']}")
    print("------------------------------")

if __name__ == "__main__":
    main()