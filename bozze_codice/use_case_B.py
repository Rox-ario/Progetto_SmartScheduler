from ortools.sat.python import cp_model

def main():
    # --- STEP 1 & 2: Import e Inizializzazione del Modello ---
    model = cp_model.CpModel()

    # --- COSTANTI DEL PROBLEMA ---
    num_standard = 13
    num_specialized = 7
    num_workers = num_standard + num_specialized
    num_days = 31     # Dal 7 Dicembre al 6 Gennaio inclusi (orizzonte specificato nel progetto)
    num_shifts = 3    # 0 = Mattina (8-14), 1 = Pomeriggio (14-20), 2 = Notte (20-8)

    # Definiamo gli insiemi di indici per distinguere i ruoli
    # I primi 13 (da 0 a 12) sono Standard, i successivi 7 (da 13 a 19) sono Specializzati
    standard_workers = list(range(num_standard))
    specialized_workers = list(range(num_standard, num_workers))

    # --- STEP 3: Variabili Decisionali ---
    # Matrice booleana: shifts[(w, d, s)] = 1 se il lavoratore 'w' lavora il giorno 'd' al turno 's'
    shifts = {}
    for w in range(num_workers):
        for d in range(num_days):
            for s in range(num_shifts):
                shifts[(w, d, s)] = model.new_bool_var(f'shift_w{w}_d{d}_s{s}')

    # --- STEP 4: Aggiunta dei Vincoli (Hard Constraints del Progetto) ---

    # Vincolo 1: Almeno 3 lavoratori per ogni turno (Requisito Caso B)
    for d in range(num_days):
        for s in range(num_shifts):
            # 1. Vincolo Specializzati: Almeno 1 specializzato per turno
            model.add(sum(shifts[(w, d, s)] for w in specialized_workers) >= 1)

            # 2. Vincolo Totale: Almeno 3 lavoratori in totale (1 Spec + 2 Std/Spec facenti funzione)
            model.add(sum(shifts[(w, d, s)] for w in range(num_workers)) >= 3)

    # Vincolo 2: Massimo 1 turno al giorno per lavoratore
    for w in range(num_workers):
        for d in range(num_days):
            model.add(sum(shifts[(w, d, s)] for s in range(num_shifts)) <= 1)

    # Vincolo 3: Due giorni liberi obbligatori dopo ogni turno di notte
    # Il turno di notte ha indice 2. Se shifts[(w, d, 2)] è vero, i giorni d+1 e d+2 devono essere vuoti.
    for w in range(num_workers):
        for d in range(num_days - 2):
            for s_next in range(num_shifts):
                # .only_enforce_if() attiva il vincolo = 0 solo se la variabile del turno di notte è a 1
                model.add(shifts[(w, d+1, s_next)] == 0).only_enforce_if(shifts[(w, d, 2)])
                model.add(shifts[(w, d+2, s_next)] == 0).only_enforce_if(shifts[(w, d, 2)])

        # Gestione del "bordo" per il penultimo giorno del mese
        if num_days >= 2:
            for s_next in range(num_shifts):
                model.add(shifts[(w, num_days-1, s_next)] == 0).only_enforce_if(shifts[(w, num_days-2, 2)])

    # Vincolo 4: Massimo 36 ore settimanali.
    # Mattina = 6h, Pomeriggio = 6h, Notte = 12h (carico doppio come da specifiche)
    # Applichiamo una finestra scorrevole di 7 giorni per garantire il rispetto su qualsiasi periodo.
    shift_hours = [6, 6, 12]
    for w in range(num_workers):
        for start_day in range(num_days - 6):
            weekly_hours = []
            for d in range(start_day, start_day + 7):
                for s in range(num_shifts):
                    weekly_hours.append(shifts[(w, d, s)] * shift_hours[s])
            model.add(sum(weekly_hours) <= 36)

    # --- STEP 5: Risoluzione (Solver) ---
    solver = cp_model.CpSolver()

    status = solver.solve(model)

    # --- Stampa dei Risultati ---
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("Soluzione FATTIBILE trovata!\n")
        shift_names = ["Mattina (8-14)", "Pomeriggio (14-20)", "Notte (20-8)"]

        for d in range(num_days):
            print(f"--- Giorno {d+1} ---")
            for s in range(num_shifts):
                workers_assigned = [w for w in range(num_workers) if solver.value(shifts[(w, d, s)]) == 1]
                print(f"  {shift_names[s]:<18}: Lavoratori {workers_assigned}")
            print()
    else:
        print("Stato Solver: INFEASIBLE. Nessuna soluzione trovata che rispetti tutti i vincoli.")

if __name__ == '__main__':
    main()