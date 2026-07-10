"""
SmartScheduler — Pipeline Runner
Unified pipeline execution for both Case A and Case B.
Provides progress callbacks for UI integration.
"""

import os
import importlib.util
from dataclasses import dataclass, field
from ortools.sat.python import cp_model

from src.system_builder import SystemBuilderAgent
from src.preferences.translator import PreferenceTranslator
from src.validation.feedback import FeedbackPromptBuilder, RefinementPromptBuilder


@dataclass
class PipelineResult:
    """Structured result from the pipeline execution."""
    success: bool = False
    schedule_dict: dict = field(default_factory=dict)
    fairness_results: dict = field(default_factory=dict)
    refinement_history: list = field(default_factory=list)
    num_workers: int = 0
    num_days: int = 31
    case_type: str = "B"
    total_iterations_building: int = 0
    total_iterations_refinement: int = 0
    error_message: str = ""
    logs: list = field(default_factory=list)
    satisfaction_weights: dict = field(default_factory=dict)


def _load_dynamic_model(model_output_path):
    """Carica dinamicamente il modulo Python generato dall'LLM."""
    spec = importlib.util.spec_from_file_location("scheduler_model", model_output_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SmartSchedulerModel


def _extract_schedule(solver, scheduler, num_workers, num_days=31):
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


def _extract_preferences(scheduler, num_workers):
    """Estrae le preferenze positive/negative dalla matrice satisfaction_weights."""
    extracted_preferences = {w: {'positive': [], 'negative': []} for w in range(num_workers)}
    for (w, d, s), weight in scheduler.satisfaction_weights.items():
        if weight > 0:
            extracted_preferences[w]['positive'].append((d, s))
        elif weight < 0:
            extracted_preferences[w]['negative'].append((d, s))
    return extracted_preferences


def _build_and_solve(model_output_path, num_workers, fairness_lower_bounds=None, use_maxmin_objective=False):
    """Costruisce e risolve il modello OR-Tools CP-SAT."""
    try:
        SmartSchedulerModel = _load_dynamic_model(model_output_path)
        scheduler = SmartSchedulerModel(num_workers=num_workers, num_days=31)

        scheduler.build_base_constraints()
        scheduler.apply_preferences()

        # Iniezione programmatica dei lower-bound di fairness
        if fairness_lower_bounds:
            for w, min_score in fairness_lower_bounds.items():
                score_w = cp_model.LinearExpr.weighted_sum(
                    [scheduler.shifts[(w, d, s)] for d in range(31) for s in range(3)],
                    [scheduler.satisfaction_weights.get((w, d, s), 0) for d in range(31) for s in range(3)]
                )
                scheduler.model.add(score_w >= min_score)

        if use_maxmin_objective:
            # Obiettivo Max-Min Fairness
            z = scheduler.model.new_int_var(-10000, 10000, 'min_satisfaction_score')
            for w in range(num_workers):
                score_w = cp_model.LinearExpr.weighted_sum(
                    [scheduler.shifts[(w, d, s)] for d in range(31) for s in range(3)],
                    [scheduler.satisfaction_weights.get((w, d, s), 0) for d in range(31) for s in range(3)]
                )
                scheduler.model.add(z <= score_w)
            scheduler.model.maximize(z)
        else:
            scheduler.build_objective()

        status, solver = scheduler.solve()
        return status, solver, scheduler
    except Exception as e:
        print(f"[-] Errore durante la costruzione/risoluzione del modello: {e}")
        return None, None, None


def _get_modules(case_type):
    """Restituisce le classi HardConstraintVerifier e FairnessEvaluationAgent corrette per il case type."""
    if case_type == "A":
        from src.validation.validation_caseA import HardConstraintVerifier, FairnessEvaluationAgent
    else:
        from src.validation.validation import HardConstraintVerifier, FairnessEvaluationAgent
    return HardConstraintVerifier, FairnessEvaluationAgent


def run_pipeline(draft_path, preferences_path, num_workers, case_type="B", log_callback=None):
    """
    Esegue la pipeline completa di SmartScheduler.

    Args:
        draft_path: Path al file model_draft.txt
        preferences_path: Path al file preferences.txt
        num_workers: Numero di lavoratori
        case_type: "A" (omogeneo) o "B" (eterogeneo)
        log_callback: function(phase: str, message: str, progress: float)
                      Chiamata ad ogni step per aggiornare la UI.

    Returns:
        PipelineResult con tutti i risultati strutturati.
    """
    model_output_path = os.path.join("src", "scheduler_model.py")
    result = PipelineResult(num_workers=num_workers, case_type=case_type)

    HardConstraintVerifier, FairnessEvaluationAgent = _get_modules(case_type)

    def log(phase, message, progress=0.0):
        result.logs.append((phase, message))
        if log_callback:
            log_callback(phase, message, progress)

    try:
        builder_agent = SystemBuilderAgent()
        translator = PreferenceTranslator()
    except ValueError as e:
        result.error_message = f"Errore di configurazione: {str(e)}"
        log("ERRORE", result.error_message, 0.0)
        return result

    MAX_RETRIES = 10
    attempt = 1
    is_schedule_valid = False
    feedback_prompt = None
    schedule_dict = {}
    scheduler = None

    # ═══════════════════════════════════════════════════════
    # CICLO FASI 0-3: Generazione e Validazione Hard Constraints
    # ═══════════════════════════════════════════════════════
    log("INIT", f"Avvio pipeline SmartScheduler — Scenario {case_type} con {num_workers} lavoratori", 0.02)

    while attempt <= MAX_RETRIES and not is_schedule_valid:
        progress_base = 0.05 + (attempt - 1) / MAX_RETRIES * 0.35

        # ── FASE 0: System Building / Revision ──
        if attempt == 1:
            log("FASE 0", "Generazione del modello base tramite LLM...", progress_base)
            success = builder_agent.generate_model_file(draft_path, model_output_path)
        else:
            log("FASE 0", f"Revisione del modello (tentativo {attempt}/{MAX_RETRIES})...", progress_base)
            success = builder_agent.generate_model_file(draft_path, model_output_path, feedback_prompt=feedback_prompt)

        if not success:
            result.error_message = "Generazione/Revisione del modello fallita dall'LLM."
            log("ERRORE", result.error_message, progress_base)
            result.total_iterations_building = attempt
            return result

        # ── FASE 1: Preferences Definition ──
        log("FASE 1", "Iniezione delle preferenze nel modello...", progress_base + 0.05)
        translator.process(preferences_path, model_output_path)

        # ── FASE 2: Schedule Drafting ──
        log("FASE 2", "Risoluzione del modello OR-Tools CP-SAT...", progress_base + 0.10)
        status, solver, scheduler = _build_and_solve(model_output_path, num_workers)

        if status is None:
            feedback_prompt = ("Il modello ha generato un errore Python durante l'esecuzione. "
                               "Rivedi la sintassi e la struttura del codice.")
            log("FASE 2", f"⚠️ Errore nell'esecuzione del modello. Tentativo {attempt}/{MAX_RETRIES}.", progress_base + 0.10)
            attempt += 1
            continue

        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            log("FASE 2", "✅ Soluzione matematica trovata!", progress_base + 0.12)
            schedule_dict = _extract_schedule(solver, scheduler, num_workers)

            # ── FASE 3: Schedule Verification ──
            log("FASE 3", "Verifica vincoli rigidi (Hard Constraints)...", progress_base + 0.15)
            verifier = HardConstraintVerifier(schedule_dict, num_workers=num_workers, num_days=31)

            if case_type == "A":
                is_valid, validation_result = verifier.verify_all(2)
            else:
                is_valid, validation_result = verifier.verify_all()

            if is_valid:
                log("FASE 3", "✅ Tutti i vincoli rigidi sono rispettati!", progress_base + 0.17)
                is_schedule_valid = True
            else:
                num_violations = len(validation_result) if isinstance(validation_result, list) else 0
                log("FASE 3", f"❌ Validazione fallita — {num_violations} violazioni trovate.", progress_base + 0.17)
                feedback_builder = FeedbackPromptBuilder()
                feedback_prompt = feedback_builder.build_revision_prompt(validation_result)
        else:
            log("FASE 2", "❌ Solver INFEASIBLE — Rilassamento vincoli...", progress_base + 0.12)
            feedback_prompt = ("Il solver ha restituito INFEASIBLE. Controlla di non aver inserito "
                               "vincoli matematicamente impossibili da soddisfare insieme. Rivedi il modello.")

        attempt += 1

    result.total_iterations_building = attempt - 1

    if not is_schedule_valid:
        result.error_message = f"Impossibile generare una schedule valida dopo {MAX_RETRIES} tentativi."
        log("ERRORE", result.error_message, 0.40)
        return result

    # ═══════════════════════════════════════════════════════
    # FASE 4: Schedule Refinement (Ciclo Iterativo di Fairness)
    # ═══════════════════════════════════════════════════════
    log("FASE 4", "Avvio ciclo di refinement fairness (max-min)...", 0.45)

    MAX_REFINEMENT = 10
    refinement_builder = RefinementPromptBuilder()
    fairness_lower_bounds = {}
    best_scheduler = scheduler
    refinement_iter = 0

    for refinement_iter in range(1, MAX_REFINEMENT + 1):
        progress_base = 0.45 + (refinement_iter - 1) / MAX_REFINEMENT * 0.45

        log("FASE 4", f"── Iterazione refinement {refinement_iter}/{MAX_REFINEMENT} ──", progress_base)

        # 4.1: Valutazione Fairness
        extracted_preferences = _extract_preferences(scheduler, num_workers)
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

        # Salvataggio nella history
        result.refinement_history.append({
            "iteration": refinement_iter,
            "min_score": current_min_score,
            "most_disadvantaged": most_disadvantaged,
            "scores": dict(satisfaction_scores),
            "std_dev": fairness_results['standard_deviation'],
            "mean_load": fairness_results['mean_disadvantaged_shifts'],
            "improved": True  # Sarà aggiornato se non migliora
        })

        log("FASE 4", f"Min score attuale: {current_min_score} (Worker {most_disadvantaged})", progress_base + 0.02)

        # 4.2-4.3: Lettura codice corrente + costruzione prompt
        with open(model_output_path, 'r', encoding='utf-8') as f:
            current_code = f.read()

        refinement_prompt = refinement_builder.build_refinement_prompt(
            most_disadvantaged_worker_id=most_disadvantaged,
            current_min_score=current_min_score,
            satisfaction_scores=satisfaction_scores,
            current_code=current_code
        )

        # 4.4: LLM rigenera il codice
        log("FASE 4", "LLM sta raffinando il modello...", progress_base + 0.05)
        success = builder_agent.generate_model_file(
            draft_path, model_output_path, feedback_prompt=refinement_prompt
        )

        if not success:
            log("FASE 4", "⚠️ LLM non è riuscito a raffinare. Fine refinement.", progress_base + 0.05)
            result.refinement_history[-1]["improved"] = False
            break

        # 4.5: Re-iniezione preferenze e risoluzione
        log("FASE 4", "Re-iniezione preferenze e risoluzione...", progress_base + 0.08)
        translator.process(preferences_path, model_output_path)

        status, solver, scheduler = _build_and_solve(
            model_output_path, num_workers,
            fairness_lower_bounds=fairness_lower_bounds,
            use_maxmin_objective=True
        )

        if status is None:
            log("FASE 4", "⚠️ Errore nell'esecuzione del modello raffinato. Fine refinement.", progress_base + 0.08)
            result.refinement_history[-1]["improved"] = False
            break

        # 4.6: Controllo risultato solver
        if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            log("FASE 4", "🏁 Solver INFEASIBLE — Fairness al massimo possibile.", progress_base + 0.08)
            result.refinement_history[-1]["improved"] = False
            break

        # 4.7: Re-verifica hard constraints
        new_schedule_dict = _extract_schedule(solver, scheduler, num_workers)
        verifier = HardConstraintVerifier(new_schedule_dict, num_workers=num_workers, num_days=31)
        if case_type == "A":
            is_valid, _ = verifier.verify_all(2)
        else:
            is_valid, _ = verifier.verify_all()

        if not is_valid:
            log("FASE 4", "❌ Nuova schedulazione viola vincoli hard. Fine refinement.", progress_base + 0.10)
            result.refinement_history[-1]["improved"] = False
            break

        # 4.8: Verifica miglioramento effettivo
        new_extracted_prefs = _extract_preferences(scheduler, num_workers)
        new_fairness_agent = FairnessEvaluationAgent(
            new_schedule_dict,
            num_workers=num_workers,
            num_days=31,
            preferences=new_extracted_prefs,
            satisfaction_weights=scheduler.satisfaction_weights
        )
        new_fairness_results = new_fairness_agent.evaluate_fairness()
        new_min_score = new_fairness_results['min_satisfaction_score']

        if new_min_score > current_min_score:
            log("FASE 4",
                f"✅ Miglioramento confermato! {current_min_score} → {new_min_score} (+{new_min_score - current_min_score})",
                progress_base + 0.12)
            schedule_dict = new_schedule_dict
            best_scheduler = scheduler

            # Aggiornamento lower-bound per la prossima iterazione
            for w, score in new_fairness_results['satisfaction_scores'].items():
                fairness_lower_bounds[w] = max(fairness_lower_bounds.get(w, score), score)
            fairness_lower_bounds[most_disadvantaged] = new_min_score + 1
        else:
            log("FASE 4", "🏁 Nessun miglioramento effettivo. Fine refinement.", progress_base + 0.12)
            result.refinement_history[-1]["improved"] = False
            break

    result.total_iterations_refinement = refinement_iter

    # ═══════════════════════════════════════════════════════
    # REPORT FINALE
    # ═══════════════════════════════════════════════════════
    log("FINE", "Generazione report finale...", 0.95)

    final_prefs = _extract_preferences(best_scheduler, num_workers)
    final_fairness = FairnessEvaluationAgent(
        schedule_dict,
        num_workers=num_workers,
        num_days=31,
        preferences=final_prefs,
        satisfaction_weights=best_scheduler.satisfaction_weights
    )
    final_results = final_fairness.evaluate_fairness()

    result.success = True
    result.schedule_dict = schedule_dict
    result.fairness_results = final_results
    result.satisfaction_weights = dict(best_scheduler.satisfaction_weights)

    log("FINE", "✅ Pipeline completata con successo!", 1.0)

    return result
