from ortools.sat.python import cp_model

class SmartSchedulerModel:
    def __init__(self, num_workers, num_days):
        self.num_workers = num_workers
        self.num_days = num_days
        self.model = cp_model.CpModel()
        self.shifts = {}
        for w in range(self.num_workers):
            for d in range(self.num_days):
                for s in range(3):
                    self.shifts[(w, d, s)] = self.model.new_bool_var(f'shift_w{w}_d{d}_s{s}')
        
        self.shift_durations = {0: 6, 1: 6, 2: 12}
        self.shift_weights = {0: 1, 1: 1, 2: 2}
        
        self.satisfaction_weights = {}
        holidays = {1, 17, 18, 25, 30}
        for w in range(self.num_workers):
            for d in range(self.num_days):
                is_weekend = (d % 7 == 5 or d % 7 == 6)
                is_holiday = d in holidays
                for s in range(3):
                    is_night = (s == 2)
                    weight = 0
                    if is_night:
                        weight -= 2
                    if is_weekend or is_holiday:
                        weight -= 1
                    self.satisfaction_weights[(w, d, s)] = weight

    def build_base_constraints(self):
        # HC1: Max 36 hours per 7-day window
        for w in range(self.num_workers):
            for d in range(self.num_days - 6):
                self.model.add(sum(self.shifts[(w, d + i, s)] * self.shift_durations[s] 
                                   for i in range(7) for s in range(3)) <= 36)

        # HC2: Exactly 25 shift weights per month
        for w in range(self.num_workers):
            self.model.add(sum(self.shifts[(w, d, s)] * self.shift_weights[s] 
                               for d in range(self.num_days) for s in range(3)) == 25)

        # HC3: Night shift rest (2 days)
        for w in range(self.num_workers):
            for d in range(self.num_days - 2):
                self.model.add_implication(self.shifts[(w, d, 2)], self.shifts[(w, d + 1, 0)].Not())
                self.model.add_implication(self.shifts[(w, d, 2)], self.shifts[(w, d + 1, 1)].Not())
                self.model.add_implication(self.shifts[(w, d, 2)], self.shifts[(w, d + 1, 2)].Not())
                self.model.add_implication(self.shifts[(w, d, 2)], self.shifts[(w, d + 2, 0)].Not())
                self.model.add_implication(self.shifts[(w, d, 2)], self.shifts[(w, d + 2, 1)].Not())
                self.model.add_implication(self.shifts[(w, d, 2)], self.shifts[(w, d + 2, 2)].Not())

        # HC4: Max one shift per day
        for w in range(self.num_workers):
            for d in range(self.num_days):
                self.model.add(sum(self.shifts[(w, d, s)] for s in range(3)) <= 1)

        # HC5: No subsequent shifts (Afternoon -> Morning)
        for w in range(self.num_workers):
            for d in range(self.num_days - 1):
                self.model.add_implication(self.shifts[(w, d, 1)], self.shifts[(w, d + 1, 0)].Not())

        # HC6: Mandatory weekly rest
        for w in range(self.num_workers):
            for d in range(self.num_days - 6):
                self.model.add(sum(self.shifts[(w, d + i, s)] for i in range(7) for s in range(3)) >= 1)

        # HC7: Min 2 workers per shift
        for d in range(self.num_days):
            for s in range(3):
                self.model.add(sum(self.shifts[(w, d, s)] for w in range(self.num_workers)) >= 2)

        # Staffing Scenario B: Specialized workers (Assuming IDs 0-5 are specialized)
        specialized = [0, 1, 2, 3, 4, 5]
        for d in range(self.num_days):
            for s in range(3):
                self.model.add(sum(self.shifts[(w, d, s)] for w in specialized) >= 1)

    def apply_preferences(self):
        """
        Metodo dedicato all'iniezione delle preferenze generate dalla Fase 1.
        Gli alias locali espongono gli attributi della classe con i nomi
        che il codice generato da Gemini si aspetta.
        NON modificare i nomi degli alias: sono il contratto con il LLM.
        """
        model = self.model
        shifts = self.shifts
        satisfaction_weights = self.satisfaction_weights

        # <<< PREFERENCES_INJECTION_POINT >>>
        # Preferenza: Worker 1 prefer morning shifts
        # Categoria 2: preferenza positiva estesa a tutti i giorni dell'orizzonte.
        for d in range(31):
            satisfaction_weights[(1, d, 0)] = 1

        # Preferenza: Worker 2 can work during weekends, but not on consecutive holidays
        # Categoria 2: preferenza positiva per i weekend.
        # Categoria 1: divieto assoluto per festività consecutive (hard constraint).
        # Giorni weekend: [5, 6, 12, 13, 19, 20, 26, 27, 30]
        # Giorni festivi: 1 (8 dic), 17 (24 dic), 18 (25 dic), 25 (1 gen), 30 (6 gen)

        # Preferenza positiva per i weekend
        weekend_days = [5, 6, 12, 13, 19, 20, 26, 27, 30]
        for d in weekend_days:
            for s in range(3):
                satisfaction_weights[(2, d, s)] = 1

        # Divieto per festività consecutive (24-25 dic)
        # Se lavora il 17 (24 dic), non può lavorare il 18 (25 dic) e viceversa.
        # Utilizziamo un vincolo di esclusione reciproca per le festività consecutive.
        for s1 in range(3):
            for s2 in range(3):
                model.add(shifts[(2, 17, s1)] + shifts[(2, 18, s2)] <= 1)

        # Preferenza: Worker 3 is available for emergency coverage twice a month
        # Categoria 3: la disponibilità generica "due volte al mese" non è mappabile su giorni specifici.

        # Preferenza: Worker 4 prefers not to work during holidays
        # Categoria 2: preferenza negativa su giorni festivi.
        # Giorni festivi nell'orizzonte: 1 (8 dic), 17 (24 dic), 18 (25 dic), 25 (1 gen), 30 (6 gen).
        for s in range(3):
            satisfaction_weights[(4, 1, s)] = -2
            satisfaction_weights[(4, 17, s)] = -2
            satisfaction_weights[(4, 18, s)] = -2
            satisfaction_weights[(4, 25, s)] = -2
            satisfaction_weights[(4, 30, s)] = -2

    def build_objective(self):
        self.model.maximize(cp_model.LinearExpr.weighted_sum(
            [self.shifts[(w, d, s)] for w in range(self.num_workers) for d in range(self.num_days) for s in range(3)],
            [self.satisfaction_weights[(w, d, s)] for w in range(self.num_workers) for d in range(self.num_days) for s in range(3)]
        ))

    def solve(self):
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 120.0
        status = solver.solve(self.model)
        return (status, solver)