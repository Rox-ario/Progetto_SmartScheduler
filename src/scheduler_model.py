from ortools.sat.python import cp_model

class SmartSchedulerModel:
    def __init__(self, num_workers, num_days):
        self.num_workers = num_workers
        self.num_days = num_days
        self.model = cp_model.CpModel()
        self.shifts = {}
        self.satisfaction_weights = {}
        self.shift_durations = {0: 6, 1: 6, 2: 12}
        self.shift_weights = {0: 1, 1: 1, 2: 2}

        for w in range(self.num_workers):
            for d in range(self.num_days):
                for s in range(3):
                    self.shifts[(w, d, s)] = self.model.new_bool_var(f'shift_w{w}_d{d}_s{s}')
                    self.satisfaction_weights[(w, d, s)] = 0

    def build_base_constraints(self):
        # HC4: Max one shift per day
        for w in range(self.num_workers):
            for d in range(self.num_days):
                self.model.add(sum(self.shifts[(w, d, s)] for s in range(3)) <= 1)

        # HC7 & HC8: Shift coverage (2 to 3 workers)
        for d in range(self.num_days):
            for s in range(3):
                self.model.add(sum(self.shifts[(w, d, s)] for w in range(self.num_workers)) >= 2)
                self.model.add(sum(self.shifts[(w, d, s)] for w in range(self.num_workers)) <= 3)

        # HC1: Rolling 7-day window (max 36 hours)
        for w in range(self.num_workers):
            for d in range(self.num_days - 6):
                self.model.add(sum(self.shifts[(w, d + i, s)] * self.shift_durations[s] 
                                   for i in range(7) for s in range(3)) <= 36)

        # HC2: Monthly workload (sum of weights = 25)
        for w in range(self.num_workers):
            self.model.add(sum(self.shifts[(w, d, s)] * self.shift_weights[s] 
                               for d in range(self.num_days) for s in range(3)) == 25)

        # HC3: Night shift rest (2 days off after night shift)
        for w in range(self.num_workers):
            for d in range(self.num_days - 2):
                self.model.add_implication(self.shifts[(w, d, 2)], self.shifts[(w, d + 1, 0)].Not())
                self.model.add_implication(self.shifts[(w, d, 2)], self.shifts[(w, d + 1, 1)].Not())
                self.model.add_implication(self.shifts[(w, d, 2)], self.shifts[(w, d + 1, 2)].Not())
                self.model.add_implication(self.shifts[(w, d, 2)], self.shifts[(w, d + 2, 0)].Not())
                self.model.add_implication(self.shifts[(w, d, 2)], self.shifts[(w, d + 2, 1)].Not())
                self.model.add_implication(self.shifts[(w, d, 2)], self.shifts[(w, d + 2, 2)].Not())

        # HC5: No subsequent shifts (Afternoon D -> Morning D+1)
        for w in range(self.num_workers):
            for d in range(self.num_days - 1):
                self.model.add_implication(self.shifts[(w, d, 1)], self.shifts[(w, d + 1, 0)].Not())

        # HC6: Mandatory weekly rest (at least one day off in every 7-day block)
        for w in range(self.num_workers):
            for d in range(self.num_days - 6):
                self.model.add(sum(sum(self.shifts[(w, d + i, s)] for s in range(3)) for i in range(7)) <= 6)

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
        # Categoria 2: preferenza positiva estesa a tutti i giorni dell'orizzonte per il lavoratore 1.
        for d in range(31):
            satisfaction_weights[(1, d, 0)] = 10

        # Preferenza: Worker 2 can work during weekends, but not on consecutive holidays
        # Categoria 2: preferenza positiva per i weekend.
        # Giorni weekend: [5, 6, 12, 13, 19, 20, 26, 27, 30]
        for d in [5, 6, 12, 13, 19, 20, 26, 27, 30]:
            for s in range(3):
                satisfaction_weights[(2, d, s)] = 10

        # Categoria 3: "non su festività consecutive" è una regola logica complessa 
        # non traducibile come semplice peso o divieto assoluto.

        # Preferenza: Worker 3 is available for emergency coverage twice a month
        # Categoria 3: disponibilità generica non legata a giorni o turni specifici.

        # Preferenza: Worker 4 prefers not to work during holidays
        # Categoria 2: preferenza negativa su giorni festivi.
        # Indici festivi nell'orizzonte: 1 (8 dic), 17 (24 dic), 18 (25 dic), 25 (1 gen), 30 (6 gen).
        for s in range(3):
            satisfaction_weights[(4, 1, s)] = -10
            satisfaction_weights[(4, 17, s)] = -10
            satisfaction_weights[(4, 18, s)] = -10
            satisfaction_weights[(4, 25, s)] = -10
            satisfaction_weights[(4, 30, s)] = -10

    def build_objective(self):
        self.model.maximize(cp_model.LinearExpr.weighted_sum(
            [self.shifts[key] for key in self.shifts],
            [self.satisfaction_weights[key] for key in self.shifts]
        ))

    def solve(self):
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 120.0
        status = solver.solve(self.model)
        return (status, solver)