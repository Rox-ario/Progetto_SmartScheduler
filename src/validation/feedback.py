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