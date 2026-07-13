"""
SmartScheduler — Interfaccia Grafica Streamlit
Entry point principale: streamlit run app.py
"""

import streamlit as st
import os
import pandas as pd
from pipeline_runner import run_pipeline
from ui.components.schedule_grid import (
    render_schedule_heatmap,
    render_schedule_table,
    get_schedule_dataframe,
)
from ui.components.fairness_charts import (
    render_satisfaction_bar_chart,
    render_fairness_radar_chart,
    render_refinement_line_chart,
    render_scores_comparison_chart,
)

# ═══════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════
st.set_page_config(
    page_title="SmartScheduler",
    page_icon="🗓️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════
# LOAD CUSTOM CSS
# ═══════════════════════════════════════════════════
def load_css():
    css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui", "styles", "theme.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


# ═══════════════════════════════════════════════════
# SESSION STATE INIT
# ═══════════════════════════════════════════════════
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None


# ═══════════════════════════════════════════════════
# SIDEBAR — Input & Configuration
# ═══════════════════════════════════════════════════
with st.sidebar:
    st.markdown("# 🗓️ SmartScheduler")
    st.markdown("*Multi-Stage AI Scheduling System*")
    st.divider()

    # ── Scenario Selection ──
    st.markdown("### 📋 Scenario")
    case_option = st.radio(
        "Seleziona lo use case",
        ["Case A — Omogeneo", "Case B — Eterogeneo"],
        index=1,
        label_visibility="collapsed",
    )
    case_type = "A" if "Case A" in case_option else "B"

    # ── Number of Workers (Fissato) ──
    num_workers = 13 if case_type == "A" else 20

    st.divider()

    # ── File Uploaders ──
    st.markdown("### 📄 File di Input")

    draft_file = st.file_uploader(
        "Model Draft (.txt)",
        type=["txt"],
        help="File con la specifica del modello: vincoli legali, turni, lavoratori.",
        key="draft_uploader",
    )

    prefs_file = st.file_uploader(
        "Preferences (.txt)",
        type=["txt"],
        help="Preferenze dei lavoratori in linguaggio naturale (una per riga).",
        key="prefs_uploader",
    )

    # ── File Previews ──
    if draft_file:
        with st.expander("Anteprima Model Draft", expanded=False):
            st.code(draft_file.getvalue().decode("utf-8"), language="text")

    if prefs_file:
        with st.expander("Anteprima Preferences", expanded=False):
            st.code(prefs_file.getvalue().decode("utf-8"), language="text")

    st.divider()

    # ── Time Window Info ──
    st.markdown("### 🕐 Arco Temporale")
    st.markdown(
        """
        <div style="
            background: rgba(22, 22, 50, 0.65);
            border: 1px solid rgba(124, 58, 237, 0.25);
            border-radius: 12px;
            padding: 12px 16px;
            font-size: 0.85rem;
            color: #a0a0c0;
        ">
            📅 <b style="color:#a78bfa">7 Dicembre 2026</b> → <b style="color:#a78bfa">6 Gennaio 2027</b><br>
            <span style="color:#6868a0">31 giorni · 3 turni/giorno · Fissa</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Run Button ──
    can_run = draft_file is not None and prefs_file is not None
    run_button = st.button(
        "▶️  Avvia Pipeline",
        type="primary",
        disabled=not can_run,
        use_container_width=True,
    )

    if not can_run:
        st.caption("⬆️ Carica entrambi i file per abilitare l'avvio.")


# ═══════════════════════════════════════════════════
# MAIN AREA — Header
# ═══════════════════════════════════════════════════
st.markdown(
    """
    <div class="main-header fade-in">
        <h1>🗓️ SmartScheduler</h1>
        <p>Framework agentico per scheduling di turni ospedalieri con AI e Constraint Programming</p>
    </div>
    """,
    unsafe_allow_html=True,
)


if run_button and can_run:
    st.session_state.pipeline_result = None

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)

    if case_type == "A":
        draft_filename = "model_draft_caseA.txt"
    else:
        draft_filename = "model_draft.txt"

    draft_path = os.path.join(data_dir, draft_filename)
    prefs_path = os.path.join(data_dir, "preferences.txt")

    with open(draft_path, "wb") as f:
        f.write(draft_file.getbuffer())
    with open(prefs_path, "wb") as f:
        f.write(prefs_file.getbuffer())


    with st.status("Pipeline in esecuzione...", expanded=True) as status:
        progress_bar = st.progress(0.0, text="Inizializzazione...")
        log_placeholder = st.empty()
        accumulated_logs = []

        def ui_log_callback(phase, message, progress):
            accumulated_logs.append(f"[{phase}] {message}")
            clamped = max(0.0, min(float(progress), 1.0))
            progress_bar.progress(clamped, text=f"**[{phase}]** {message}")
            log_placeholder.code("\n".join(accumulated_logs[-25:]), language="text")

        try:
            result = run_pipeline(
                draft_path=draft_path,
                preferences_path=prefs_path,
                num_workers=num_workers,
                case_type=case_type,
                log_callback=ui_log_callback,
            )

            st.session_state.pipeline_result = result

            if result.success:
                progress_bar.progress(1.0, text="Pipeline completata!")
                status.update(label="EVVAI! Pipeline completata con successo!", state="complete", expanded=False)
            else:
                status.update(
                    label=f"OH NO! Pipeline fallita: {result.error_message}",
                    state="error",
                    expanded=True,
                )

        except Exception as e:
            status.update(label=f"Errore critico: {str(e)}", state="error", expanded=True)
            st.error(f"**Errore durante l'esecuzione della pipeline:**\n\n`{str(e)}`")


result = st.session_state.get("pipeline_result")

if result and result.success:
    st.divider()

    # ── Tabs dei risultati ──
    tab_calendar, tab_fairness, tab_report, tab_logs = st.tabs([
        "📅 Calendario",
        "📊 Fairness",
        "📋 Report",
        "📜 Logs",
    ])

    with tab_calendar:
        st.markdown("### 📅 Calendario Turni Generato")

        # Filtro per lavoratore
        col_filter, col_view = st.columns([1, 3])

        with col_filter:
            worker_options = ["👥 Tutti i lavoratori"] + [f"Worker {w}" for w in range(result.num_workers)]
            selected_option = st.selectbox(
                "🔍 Filtra per lavoratore",
                worker_options,
                help="Seleziona un lavoratore per evidenziare i suoi turni.",
            )
            selected_worker = (
                None if selected_option.startswith("👥")
                else int(selected_option.split(" ")[1])
            )

            # Selettore vista
            view_mode = st.radio(
                "Modalità vista",
                ["🗺️ Heatmap", "📊 Tabella"],
                horizontal=True,
                label_visibility="collapsed",
            )

        with col_view:
            if "Heatmap" in view_mode:
                render_schedule_heatmap(
                    result.schedule_dict,
                    result.num_workers,
                    selected_worker=selected_worker,
                )
            else:
                render_schedule_table(result.schedule_dict)

        st.divider()
        st.markdown("### 📥 Download")

        dl_col1, dl_col2, dl_col3 = st.columns(3)

        df = get_schedule_dataframe(result.schedule_dict)
        csv_data = df.to_csv(index=False, encoding="utf-8")
        dl_col1.download_button(
            "📥 Scarica CSV",
            data=csv_data,
            file_name="smartscheduler_calendario.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # Excel Download
        try:
            from io import BytesIO
            excel_buffer = BytesIO()
            df.to_excel(excel_buffer, index=False, engine="openpyxl")
            excel_data = excel_buffer.getvalue()
            dl_col2.download_button(
                "📥 Scarica Excel",
                data=excel_data,
                file_name="smartscheduler_calendario.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except ImportError:
            dl_col2.info("Installa `openpyxl` per il download Excel.")

        import json
        json_schedule = {}
        for d, shifts in result.schedule_dict.items():
            json_schedule[str(d)] = {str(s): workers for s, workers in shifts.items()}
        json_data = json.dumps(json_schedule, indent=2, ensure_ascii=False)
        dl_col3.download_button(
            "📥 Scarica JSON",
            data=json_data,
            file_name="smartscheduler_calendario.json",
            mime="application/json",
            use_container_width=True,
        )

    with tab_fairness:
        st.markdown("### 📊 Analisi della Fairness")

        fr = result.fairness_results

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric(
            "Media Turni Disagiati",
            f"{fr['mean_disadvantaged_shifts']:.2f}",
        )
        m_col2.metric(
            "Deviazione Standard",
            f"{fr['standard_deviation']:.2f}",
        )
        m_col3.metric(
            "Worker Più Svantaggiato",
            f"Worker {fr['most_disadvantaged_worker_id']}",
        )
        m_col4.metric(
            "Min Satisfaction Score",
            fr['min_satisfaction_score'],
        )

        st.divider()

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            render_satisfaction_bar_chart(
                fr['satisfaction_scores'],
                fr['most_disadvantaged_worker_id'],
            )

        with chart_col2:
            render_fairness_radar_chart(
                fr['worker_metrics'],
                result.num_workers,
            )

    with tab_report:
        st.markdown("### 📋 Report Completo")

        info_col1, info_col2 = st.columns(2)

        with info_col1:
            st.markdown(
                f"""
                <div style="
                    background: rgba(22, 22, 50, 0.65);
                    border: 1px solid rgba(124, 58, 237, 0.25);
                    border-radius: 12px;
                    padding: 20px;
                ">
                    <h4 style="color:#a78bfa; margin-top:0;">⚙️ Configurazione</h4>
                    <table style="width:100%; color:#a0a0c0; font-size:0.9rem;">
                        <tr><td>Scenario</td><td style="text-align:right"><b style="color:#e8e8f4">Case {result.case_type}</b></td></tr>
                        <tr><td>Lavoratori</td><td style="text-align:right"><b style="color:#e8e8f4">{result.num_workers}</b></td></tr>
                        <tr><td>Giorni</td><td style="text-align:right"><b style="color:#e8e8f4">{result.num_days}</b></td></tr>
                        <tr><td>Finestra</td><td style="text-align:right"><b style="color:#e8e8f4">7 Dic 2026 – 6 Gen 2027</b></td></tr>
                    </table>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with info_col2:
            st.markdown(
                f"""
                <div style="
                    background: rgba(22, 22, 50, 0.65);
                    border: 1px solid rgba(124, 58, 237, 0.25);
                    border-radius: 12px;
                    padding: 20px;
                ">
                    <h4 style="color:#a78bfa; margin-top:0;">📊 Esecuzione</h4>
                    <table style="width:100%; color:#a0a0c0; font-size:0.9rem;">
                        <tr><td>Iterazioni Building</td><td style="text-align:right"><b style="color:#e8e8f4">{result.total_iterations_building}</b></td></tr>
                        <tr><td>Iterazioni Refinement</td><td style="text-align:right"><b style="color:#e8e8f4">{result.total_iterations_refinement}</b></td></tr>
                        <tr><td>Stato</td><td style="text-align:right"><b style="color:#10b981">Successo</b></td></tr>
                        <tr><td>Log entries</td><td style="text-align:right"><b style="color:#e8e8f4">{len(result.logs)}</b></td></tr>
                    </table>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()

        # Tabella satisfaction scores
        st.markdown("#### Satisfaction Scores Dettagliati")
        fr = result.fairness_results
        scores_data = []
        for w in sorted(fr['satisfaction_scores'].keys()):
            score = fr['satisfaction_scores'][w]
            metrics = fr['worker_metrics'].get(w, {})
            is_worst = w == fr['most_disadvantaged_worker_id']
            scores_data.append({
                "Lavoratore": f"Worker {w}" + (" ⚠️" if is_worst else ""),
                "Satisfaction Score": score,
                "Turni Disagiati": metrics.get('disadvantaged_shifts', 0),
                "Preferenze Soddisfatte": metrics.get('preference_score', 0),
            })

        scores_df = pd.DataFrame(scores_data)
        st.dataframe(scores_df, use_container_width=True, hide_index=True)

    # ─────────────────────────────────
    # TAB 4: Logs
    # ─────────────────────────────────
    with tab_logs:
        st.markdown("### 📜 Log di Esecuzione")

        if result.logs:
            # Filtraggio per fase
            all_phases = sorted(set(phase for phase, _ in result.logs))
            selected_phases = st.multiselect(
                "Filtra per fase",
                all_phases,
                default=all_phases,
                help="Seleziona le fasi da visualizzare.",
            )

            filtered_logs = [
                (phase, msg) for phase, msg in result.logs
                if phase in selected_phases
            ]

            # Formattazione log
            log_lines = []
            for phase, msg in filtered_logs:
                log_lines.append(f"[{phase:8s}] {msg}")

            st.code("\n".join(log_lines), language="text")

            st.caption(f"Totale: {len(filtered_logs)} / {len(result.logs)} log entries")
        else:
            st.info("Nessun log disponibile.")


elif result and not result.success:
    # ── Pipeline fallita ──
    st.divider()
    st.error(f"### Pipeline fallita\n\n{result.error_message}")

    if result.logs:
        with st.expander("📜 Mostra log di esecuzione", expanded=False):
            for phase, msg in result.logs:
                st.text(f"[{phase}] {msg}")

else:
    # ═══════════════════════════════════════════════════
    # WELCOME SCREEN
    # ═══════════════════════════════════════════════════

    # Info banner
    st.markdown(
        """
        <div class="info-banner fade-in">
            <span class="info-icon">👈</span>
            <p>Carica i file di input dalla <b>sidebar</b> e clicca <b>Avvia Pipeline</b> per iniziare.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Welcome cards
    st.markdown(
        """
        <div class="welcome-grid fade-in">
            <div class="welcome-card">
                <span class="card-icon">📄</span>
                <h3>Model Draft</h3>
                <p>File di specifica con vincoli legali, turni disponibili, tipologia di lavoratori e requisiti di copertura.</p>
            </div>
            <div class="welcome-card">
                <span class="card-icon">💬</span>
                <h3>Preferences</h3>
                <p>Preferenze dei lavoratori espresse in linguaggio naturale. L'AI le traduce automaticamente in vincoli formali.</p>
            </div>
            <div class="welcome-card">
                <span class="card-icon">🤖</span>
                <h3>Pipeline AI</h3>
                <p>Gemini genera il modello OR-Tools, il solver trova la soluzione ottima, e un ciclo di fairness la raffina.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Spiegazione fasi
    st.markdown("---")
    st.markdown("### Come funziona SmartScheduler")

    phase_col1, phase_col2 = st.columns(2)

    with phase_col1:
        st.markdown(
            """
            **🔧 Fase 0 — System Building**
            > L'LLM (Gemini) legge il model draft e genera un modello
            > OR-Tools CP-SAT completo in Python.

            **💬 Fase 1 — Preferences Definition**
            > Le preferenze in linguaggio naturale vengono tradotte in
            > vincoli soft (pesi di soddisfazione) e hard (indisponibilità).
            """
        )

    with phase_col2:
        st.markdown(
            """
            **📐 Fasi 2-3 — Drafting & Verification**
            > Il solver trova una soluzione ottima. Un verificatore simbolico
            > controlla tutti i vincoli legali. Se fallisce, si ripete.

            **⚖️ Fase 4 — Fairness Refinement**
            > Un ciclo iterativo migliora il satisfaction score del lavoratore
            > più svantaggiato senza peggiorare gli altri (max-min fairness).
            """
        )
