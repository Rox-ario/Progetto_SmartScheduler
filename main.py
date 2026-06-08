from src.preferences.translator import PreferenceTranslator

def main():
    # Assicurati di avere GEMINI_API_KEY nelle variabili d'ambiente
    translator = PreferenceTranslator()

    test_inputs = [
        "Il lavoratore 4 non può fare il turno di pomeriggio il giorno 10.",
        "Il lavoratore 1 vorrebbe fare la mattina nei giorni 12 e 13 se possibile.",
        "Marco lavora spesso il martedì." # Questo dovrebbe triggerare l'AMBIGUOUS
    ]

    for text in test_inputs:
        print(f"INPUT: {text}")
        code_output = translator.translate_preference(text)
        print(f"CODICE GENERATO:\n{code_output}\n")
        print("-" * 40)

if __name__ == "__main__":
    main()