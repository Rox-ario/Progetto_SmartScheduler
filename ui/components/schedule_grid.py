"""
SmartScheduler — Schedule Grid Component
Heatmap interattiva del calendario turni con Plotly.
"""

import plotly.graph_objects as go
import pandas as pd
import streamlit as st
from datetime import date, timedelta


# ── Costanti di mappatura ──
SHIFT_NAMES = {0: "Mattina (08-14)", 1: "Pomeriggio (14-20)", 2: "Notte (20-08)"}
SHIFT_SHORT = {0: "Mattina", 1: "Pomeriggio", 2: "Notte"}
START_DATE = date(2026, 12, 7)

# Giorni festivi nell'orizzonte (indici 0-based)
HOLIDAYS = {1: "8 Dic", 17: "24 Dic", 18: "25 Dic (Natale)", 25: "1 Gen (Capodanno)", 30: "6 Gen (Epifania)"}


def _day_label(day_index):
    """Genera l'etichetta per un giorno dell'orizzonte."""
    d = START_DATE + timedelta(days=day_index)
    day_names = ["Lun", "Mar", "Mer", "Gio", "Ven", "Sab", "Dom"]
    day_name = day_names[d.weekday()]
    holiday_tag = f" 🎄" if day_index in HOLIDAYS else ""
    return f"G{day_index+1} — {day_name} {d.strftime('%d/%m')}{holiday_tag}"


def _is_weekend(day_index):
    """Verifica se un giorno è sabato o domenica."""
    return day_index % 7 == 5 or day_index % 7 == 6


def render_schedule_heatmap(schedule_dict, num_workers, num_days=31, selected_worker=None):
    """
    Renderizza la heatmap interattiva del calendario turni.

    Args:
        schedule_dict: {day: {shift: [worker_ids]}}
        num_workers: Numero totale di lavoratori
        num_days: Numero di giorni nell'orizzonte
        selected_worker: Se specificato, evidenzia solo i turni di questo worker
    """
    # Preparazione dati per la heatmap
    z_values = []     # Valori numerici per il colore
    hover_texts = []  # Testo del tooltip
    y_labels = []     # Etichette asse Y (giorni)

    for d in range(num_days):
        row_z = []
        row_hover = []
        y_labels.append(_day_label(d))

        for s in range(3):
            workers = schedule_dict.get(d, {}).get(s, [])

            if selected_worker is not None:
                # Modalità filtro: evidenzia solo il worker selezionato
                if selected_worker in workers:
                    row_z.append(3)  # Evidenziato
                    row_hover.append(
                        f"<b>{SHIFT_NAMES[s]}</b><br>"
                        f"<b>✅ Worker {selected_worker} ASSEGNATO</b><br>"
                        f"Totale: {len(workers)} lavoratori<br>"
                        f"Team: {', '.join(f'W{w}' for w in workers)}"
                    )
                else:
                    row_z.append(0)  # Non assegnato
                    row_hover.append(
                        f"<b>{SHIFT_NAMES[s]}</b><br>"
                        f"Worker {selected_worker} non assegnato<br>"
                        f"Totale: {len(workers)} lavoratori"
                    )
            else:
                # Modalità globale: colore basato su numero di worker
                count = len(workers)
                # Peso visivo: notti e weekend più scuri
                visual_weight = count
                if s == 2:
                    visual_weight += 0.5  # Notte leggermente più intensa
                if _is_weekend(d) or d in HOLIDAYS:
                    visual_weight += 0.3  # Weekend/festivi

                row_z.append(visual_weight)
                workers_str = ', '.join(f'W{w}' for w in workers) if workers else 'Nessuno'

                # Tag speciale per tipologia
                shift_type = "🌙 " if s == 2 else ("☀️ " if s == 0 else "🌅 ")
                day_type = " 📅 Weekend" if _is_weekend(d) else ""
                day_type += f" 🎄 {HOLIDAYS[d]}" if d in HOLIDAYS else ""

                row_hover.append(
                    f"<b>{shift_type}{SHIFT_NAMES[s]}</b>{day_type}<br>"
                    f"<b>{count} lavoratori</b><br>"
                    f"{workers_str}"
                )

        z_values.append(row_z)
        hover_texts.append(row_hover)

    # ── Costruzione Heatmap Plotly ──
    if selected_worker is not None:
        # Colorscale per modalità filtro worker
        colorscale = [
            [0.0, "rgba(30, 30, 60, 0.3)"],   # Non assegnato
            [0.5, "rgba(30, 30, 60, 0.3)"],
            [0.5, "#7c3aed"],                   # Assegnato
            [1.0, "#a78bfa"]
        ]
    else:
        # Colorscale blu/viola per modalità globale
        colorscale = [
            [0.0, "rgba(15, 15, 35, 0.8)"],
            [0.2, "#1e1b4b"],
            [0.4, "#312e81"],
            [0.6, "#4c1d95"],
            [0.8, "#6d28d9"],
            [1.0, "#8b5cf6"]
        ]

    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=[SHIFT_SHORT[s] for s in range(3)],
        y=y_labels,
        hovertext=hover_texts,
        hoverinfo="text",
        colorscale=colorscale,
        showscale=False,
        xgap=3,
        ygap=2
    ))

    title = (f"Turni assegnati a Worker {selected_worker}"
             if selected_worker is not None
             else "Calendario Turni — Panoramica")

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color="#e8e8f4", family="Inter"),
            x=0.5, xanchor="center",
            y=0.98,
            yanchor="top"
        ),
        xaxis=dict(
            # CORREZIONE: Incapsuliamo text e font dentro il dizionario title
            title=dict(
                text="Turno",
                font=dict(size=14, color="#a0a0c0", family="Inter"),
                standoff=15
            ),
            side="top",
            tickfont=dict(size=13, color="#a0a0c0", family="Inter"),
        ),
        yaxis=dict(
            # CORREZIONE: Applichiamo la stessa logica anche all'asse Y
            title=dict(text=""),
            autorange="reversed",
            tickfont=dict(size=11, color="#a0a0c0", family="Inter"),
            dtick=1,
        ),
        height=max(600, num_days * 26),
        margin=dict(l=160, r=30, t=110, b=30),
        plot_bgcolor="rgba(10, 10, 26, 0.9)",
        paper_bgcolor="rgba(10, 10, 26, 0.0)",
        font=dict(family="Inter"),
    )

    # Annotazioni con il conteggio worker in ogni cella
    for d in range(num_days):
        for s in range(3):
            workers = schedule_dict.get(d, {}).get(s, [])
            count = len(workers)

            if selected_worker is not None:
                # Modalità filtro: mostra ✓ o —
                display_text = "✓" if selected_worker in workers else "—"
                text_color = "#a78bfa" if selected_worker in workers else "rgba(100,100,140,0.4)"
            else:
                display_text = str(count) if count > 0 else "—"
                text_color = "#e8e8f4" if count > 0 else "rgba(100,100,140,0.4)"

            fig.add_annotation(
                x=SHIFT_SHORT[s],
                y=y_labels[d],
                text=display_text,
                showarrow=False,
                font=dict(size=12, color=text_color, family="Inter", weight=600 if count > 0 else 400),
            )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def get_schedule_dataframe(schedule_dict, num_workers=None, num_days=31):
    """
    Converte il dizionario schedule in un DataFrame pandas per il download.

    Returns:
        DataFrame con colonne: Giorno, Data, Mattina, Pomeriggio, Notte
    """
    rows = []
    for d in range(num_days):
        day_date = START_DATE + timedelta(days=d)
        day_names = ["Lunedì", "Martedì", "Mercoledì", "Giovedì", "Venerdì", "Sabato", "Domenica"]

        row = {
            "Giorno": d + 1,
            "Data": day_date.strftime("%d/%m/%Y"),
            "Giorno Settimana": day_names[day_date.weekday()],
            "Festivo": HOLIDAYS.get(d, ""),
        }

        for s in range(3):
            workers = schedule_dict.get(d, {}).get(s, [])
            row[SHIFT_NAMES[s]] = ", ".join(f"W{w}" for w in workers) if workers else "—"
            row[f"N. Lavoratori ({SHIFT_SHORT[s]})"] = len(workers)

        rows.append(row)

    return pd.DataFrame(rows)


def render_schedule_table(schedule_dict, num_days=31):
    """
    Renderizza una tabella dettagliata navigabile con st.dataframe.
    """
    df = get_schedule_dataframe(schedule_dict, num_days=num_days)

    # Colonne da mostrare nella tabella UI
    display_cols = ["Giorno", "Data", "Giorno Settimana", "Festivo",
                    "Mattina (08-14)", "Pomeriggio (14-20)", "Notte (20-08)"]

    st.dataframe(
        df[display_cols],
        use_container_width=True,
        height=600,
        hide_index=True,
    )
