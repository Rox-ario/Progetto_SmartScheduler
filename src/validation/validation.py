import math


class HardConstraintVerifier:
    def __init__(self, schedule, num_workers, num_days=31):
        """
        schedule: dict strutturato come schedule[day][shift] = [worker_id_1, worker_id_2, ...]
        day: 0..30 (rappresenta i 31 giorni)
        shift: 0 (Mattina, 6h), 1 (Pomeriggio, 6h), 2 (Notte, 12h)
        """
        self.schedule = schedule
        self.num_workers = num_workers
        self.num_days = num_days
        self.shift_hours = {0: 6, 1: 6, 2: 12}
        self.errors = []

    def _get_worker_shifts(self):
        """Estrae i turni per ogni lavoratore sotto forma di coordinate (day, shift)."""
        worker_shifts = {w: [] for w in range(self.num_workers)}
        for d in range(self.num_days):
            if d in self.schedule:
                for s in range(3):
                    if s in self.schedule[d]:
                        for w in self.schedule[d][s]:
                            worker_shifts[w].append((d, s))

        # Ordiniamo cronologicamente i turni per ogni lavoratore
        for w in worker_shifts:
            worker_shifts[w].sort(key=lambda x: (x[0], x[1]))
        return worker_shifts

    def verify_all(self, min_workers_per_shift):
        worker_shifts = self._get_worker_shifts()
        self._check_shift_rules(worker_shifts)
        self._check_legal_work_limits(worker_shifts)
        self._check_staffing_requirements(min_workers_per_shift)

        if not self.errors:
            return True, "Tutti i vincoli rigidi (Hard Constraints) sono rispettati."
        return False, self.errors

    def _check_shift_rules(self, worker_shifts):
        """Verifica max 1 turno/giorno, divieto turni consecutivi, e riposo post-notte."""
        for w, shifts in worker_shifts.items():
            for i in range(len(shifts) - 1):
                d1, s1 = shifts[i]
                d2, s2 = shifts[i+1]

                # 1. Max 1 turno al giorno
                if d1 == d2:
                    self.errors.append(f"Worker {w} ha più di un turno il giorno {d1}.")

                # 2. Divieto turni consecutivi (matematicamente: differenza di indice assoluto pari a 1)
                abs_shift_1 = d1 * 3 + s1
                abs_shift_2 = d2 * 3 + s2
                if abs_shift_2 - abs_shift_1 == 1:
                    self.errors.append(f"Worker {w} ha turni consecutivi tra {d1}-T{s1} e {d2}-T{s2}.")

                # 3. Riposo dopo la notte (2 giorni liberi)
                if s1 == 2 and d2 <= d1 + 2:
                    self.errors.append(f"Worker {w} non ha avuto 2 giorni di riposo dopo la notte del giorno {d1}.")

    def _check_legal_work_limits(self, worker_shifts):
        """Verifica max 25 turni/mese e 36h/settimana."""
        for w, shifts in worker_shifts.items():
            # Max 25 turni
            if len(shifts) > 25:
                self.errors.append(f"Worker {w} eccede il limite mensile: {len(shifts)} turni.")

            # Max 36 ore su finestra mobile di 7 giorni
            for start_day in range(self.num_days - 6):
                end_day = start_day + 6
                weekly_hours = sum(
                    self.shift_hours[s] for d, s in shifts if start_day <= d <= end_day
                )
                if weekly_hours > 36:
                    self.errors.append(
                        f"Worker {w} eccede le 36h ({weekly_hours}h) nella settimana {start_day}-{end_day}."
                    )

    def _check_staffing_requirements(self, min_workers):
        """Verifica che ogni turno rispetti la copertura minima richiesta (HC7)."""
        for d in range(self.num_days):
            for s in range(3):
                workers_in_shift = self.schedule.get(d, {}).get(s, []) # Se 'd' non esiste, restituisce {}; se 's' non esiste, restituisce [].
                num_workers = len(workers_in_shift)

                # Controllo Under-staffing (copertura minima)
                if num_workers < min_workers:
                    self.errors.append(
                        f"Violazione Staffing (Under-staffing): Giorno {d}, Turno {s} "
                        f"ha {num_workers} lavoratori (minimo: {min_workers})."
                    )

class FairnessEvaluationAgent:
    def __init__(self, schedule, num_workers, num_days=31, preferences=None, satisfaction_weights=None):
        self.schedule = schedule
        self.num_workers = num_workers
        self.num_days = num_days
        # preferences è un dict: {worker_id: {'positive': [(day, shift), ...], 'negative': [...]}}
        self.preferences = preferences if preferences else {}
        # satisfaction_weights è il dizionario {(w, d, s): int} dal modello OR-Tools
        self.satisfaction_weights = satisfaction_weights if satisfaction_weights else {}

    def _calculate_base_metrics(self):
        """Calcola i turni disagiati e il punteggio preferenze per ogni lavoratore."""
        metrics = {w: {'disadvantaged_shifts': 0, 'preference_score': 0} for w in range(self.num_workers)}

        for d in range(self.num_days):
            is_weekend = (d % 7 == 5 or d % 7 == 6)
            for s in range(3):
                is_night = (s == 2)
                workers = self.schedule.get(d, {}).get(s, [])

                for w in workers:
                    # Estraiamo le preferenze del lavoratore (di default liste vuote se non ne ha)
                    worker_prefs = self.preferences.get(w, {'positive': [], 'negative': []})

                    is_desired = (d, s) in worker_prefs.get('positive', [])
                    is_undesired = (d, s) in worker_prefs.get('negative', [])

                    if is_desired:
                        # 1. Se lo voleva, sale SOLO la soddisfazione
                        metrics[w]['preference_score'] += 1
                    elif is_undesired:
                        # 2. Se NON lo voleva, è un malus enorme. Aggiungiamo 2 punti di disagio base.
                        metrics[w]['disadvantaged_shifts'] += 2
                        # Se in più è notte o weekend, il disagio si accumula ulteriormente
                        if is_weekend or is_night:
                            metrics[w]['disadvantaged_shifts'] += 0.5
                    else:
                        # 3. Turno neutro (non espresso nelle preferenze): logica standard
                        if is_weekend or is_night:
                            metrics[w]['disadvantaged_shifts'] += 1
                        if is_weekend and is_night:
                            metrics[w]['disadvantaged_shifts'] += 0.5

        return metrics

    def _calculate_satisfaction_scores(self):
        """
        Calcola il satisfaction_score intero per ogni lavoratore.
        Usa la stessa formula della funzione obiettivo di OR-Tools:
            score(w) = Σ satisfaction_weights[(w, d, s)] * assignment(w, d, s)
        dove assignment vale 1 se il lavoratore w è assegnato al turno (d, s), 0 altrimenti.

        Questo punteggio è direttamente confrontabile tra iterazioni e può essere
        usato come lower-bound nei vincoli hard della Fase 4 senza problemi di float.
        """
        scores = {}
        for w in range(self.num_workers):
            total = 0
            for d in range(self.num_days):
                for s in range(3):
                    # Il lavoratore è assegnato a questo turno?
                    is_assigned = w in self.schedule.get(d, {}).get(s, [])
                    if is_assigned:
                        weight = self.satisfaction_weights.get((w, d, s), 0)
                        total += weight
            scores[w] = total
        return scores

    def evaluate_fairness(self):
        """Identifica squilibri calcolando media, varianza e deviazione standard."""
        metrics = self._calculate_base_metrics()

        # Estraiamo l'array dei carichi disagiati
        disadvantaged_loads = [metrics[w]['disadvantaged_shifts'] for w in range(self.num_workers)]

        # Matematica: Calcolo della Media (μ)
        mean_load = sum(disadvantaged_loads) / self.num_workers

        # Matematica: Calcolo della Varianza (σ²) e Deviazione Standard (σ)
        # Varianza = Σ(x - μ)² / N
        variance = sum((x - mean_load) ** 2 for x in disadvantaged_loads) / self.num_workers
        std_dev = math.sqrt(variance)

        # ── Satisfaction Score (intero, replicabile in OR-Tools) ──
        satisfaction_scores = self._calculate_satisfaction_scores()

        # Il lavoratore più svantaggiato è quello con il satisfaction_score più basso
        min_score = min(satisfaction_scores.values()) if satisfaction_scores else 0
        most_disadvantaged = min(satisfaction_scores, key=satisfaction_scores.get)

        return {
            "mean_disadvantaged_shifts": round(mean_load, 2),
            "standard_deviation": round(std_dev, 2),
            "most_disadvantaged_worker_id": most_disadvantaged,
            "worker_metrics": metrics,
            "satisfaction_scores": satisfaction_scores,
            "min_satisfaction_score": min_score
        }