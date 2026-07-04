class FeedbackPromptBuilder:
    def __init__(self, system_context="SmartScheduler CP-SAT Model"):
        self.system_context = system_context

    def build_revision_prompt(self, errors):
        """
        Genera il prompt in linguaggio naturale da inviare al Drafting Agent
        basandosi sulla lista di errori restituita dall'HardConstraintVerifier.
        """
        if not errors:
            return "Nessun errore rilevato. La bozza è valida."

        # Intestazione del prompt per contestualizzare l'LLM
        prompt = f"""
                Sei lo Schedule Drafting Agent (Esperto in Python e Google OR-Tools CP-SAT).
                L'ultima esecuzione del tuo modello per lo {self.system_context} ha generato una turnazione che FALLISCE la validazione formale.
                
                Il modulo di verifica simbolica ha rilevato le seguenti violazioni dei vincoli rigidi (Hard Constraints):
                
                """
        # Formattazione strutturata degli errori
        for idx, error in enumerate(errors, 1):
            prompt += f"{idx}. VIOLAZIONE: {error}\n"

        # Istruzioni operative rigide per la correzione del codice
        prompt += """
                IL TUO COMPITO:
                Devi correggere il codice Python del modello OR-Tools (Fase 2) per impedire che queste violazioni si ripetano. 
                
                Linee guida per la correzione:
                1. Analizza ogni violazione e traducila nel corrispondente vincolo matematico usando `model.add(...)`.
                2. Controlla le somme logiche: per il limite delle 36 ore settimanali, usa `sum()` sulle ore dei turni assegnati nella finestra mobile di 7 giorni.
                3. Controlla gli indici: assicurati di usare le matrici booleane corrette (es. `shifts[(w, d, s)]`).
                4. Per le violazioni di Staffing, verifica che i vincoli di `min_workers` e `max_workers` siano applicati ad ogni singolo turno `(d, s)`.
                5. NON rimuovere o alterare i vincoli preesistenti che già funzionavano.
                
                Fornisci esclusivamente il blocco di codice Python aggiornato (Import, Model, Variables, Constraints, Solver) senza codice fittizio o troncato. Spiega brevemente quale vincolo hai aggiunto o corretto.
                """
        return prompt


class RefinementPromptBuilder:
    """
    Costruisce il prompt per la Fase 4 (Schedule Refinement).

    L'agente di verifica simbolico (FairnessEvaluationAgent) ha identificato
    il lavoratore più svantaggiato e calcolato i satisfaction_score di tutti.
    Questo builder compone il prompt che chiede al Drafting Agent (LLM) di
    aggiungere vincoli hard per alzare il punteggio minimo.
    """
    def __init__(self, system_context="SmartScheduler CP-SAT Model"):
        self.system_context = system_context

    def build_refinement_prompt(
        self,
        most_disadvantaged_worker_id,
        current_min_score,
        satisfaction_scores,
        current_code
    ):
        """
        Genera il prompt di refinement per il Drafting Agent.

        Args:
            most_disadvantaged_worker_id: ID del lavoratore con lo score più basso.
            current_min_score: Il satisfaction_score attuale del lavoratore più svantaggiato.
            satisfaction_scores: Dizionario {worker_id: score} di tutti i lavoratori.
            current_code: Il codice Python corrente del modello OR-Tools.

        Returns:
            Il prompt completo da inviare al Drafting Agent.
        """
        # Formattazione dei punteggi per contesto
        scores_summary = "\n".join(
            f"  - Worker {w}: satisfaction_score = {score}"
            for w, score in sorted(satisfaction_scores.items())
        )

        prompt = f"""Sei lo Schedule Drafting Agent (Esperto in Python e Google OR-Tools CP-SAT).
L'agente di verifica simbolica della fairness ha analizzato la schedulazione corrente
generata dal tuo modello per lo {self.system_context}.

═══════════════════════════════════════
RISULTATI DELLA VALUTAZIONE DI FAIRNESS
═══════════════════════════════════════

Satisfaction score attuale di ogni lavoratore (calcolato come somma pesata dei
satisfaction_weights per i turni assegnati — la stessa formula usata dalla
funzione obiettivo del modello):

{scores_summary}

Il lavoratore più svantaggiato è: Worker {most_disadvantaged_worker_id}
Il suo satisfaction_score attuale è: {current_min_score}

═══════════════════════════════════════
IL TUO COMPITO
═══════════════════════════════════════

Devi raffinare il modello OR-Tools per migliorare la fairness della schedulazione.
Per farlo, aggiungi i seguenti DUE vincoli hard al metodo `build_base_constraints()`
(o in un punto equivalente del modello, PRIMA della chiamata al solver):

1. VINCOLO FAIRNESS SUL LAVORATORE SVANTAGGIATO:
   Il satisfaction_score del Worker {most_disadvantaged_worker_id} nel nuovo schedule
   deve essere STRETTAMENTE MAGGIORE di {current_min_score}.

   Implementazione: calcola la somma pesata
       score_w = Σ satisfaction_weights[(w, d, s)] * shifts[(w, d, s)]
   per w = {most_disadvantaged_worker_id}, e aggiungi:
       model.Add(score_w >= {current_min_score + 1})

2. VINCOLO DI NON-PEGGIORAMENTO PER TUTTI GLI ALTRI LAVORATORI:
   Il satisfaction_score di OGNI altro lavoratore deve rimanere >= {current_min_score}
   (il minimo attuale), in modo che il miglioramento del lavoratore svantaggiato
   non venga ottenuto a spese degli altri.

   Implementazione: per ogni w != {most_disadvantaged_worker_id}, calcola score_w
   e aggiungi:
       model.Add(score_w >= {current_min_score})

═══════════════════════════════════════
REGOLE CRITICHE
═══════════════════════════════════════

- NON modificare la matrice satisfaction_weights. I pesi devono restare identici.
- NON rimuovere o alterare i vincoli preesistenti (hard constraints legali).
- NON modificare la struttura della classe (metodi, __init__, apply_preferences, ecc.).
- Il marcatore `# <<< PREFERENCES_INJECTION_POINT >>>` DEVE restare intatto.
- Il solver troverà autonomamente una nuova assegnazione dei turni che soddisfa
  i nuovi vincoli di fairness mantenendo tutti i vincoli legali.

═══════════════════════════════════════
CODICE CORRENTE DEL MODELLO
═══════════════════════════════════════

{current_code}

Restituisci ESCLUSIVAMENTE il codice Python aggiornato e completo della classe
SmartSchedulerModel, pronto per essere eseguito."""

        return prompt