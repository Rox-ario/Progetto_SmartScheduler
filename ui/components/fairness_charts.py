

import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import math


COLORS = {
    "violet": "#7c3aed",
    "violet_light": "#a78bfa",
    "blue": "#3b82f6",
    "blue_light": "#60a5fa",
    "cyan": "#06b6d4",
    "emerald": "#10b981",
    "rose": "#f43f5e",
    "amber": "#f59e0b",
    "bg": "rgba(10, 10, 26, 0.0)",
    "grid": "rgba(100, 100, 160, 0.15)",
    "text": "#a0a0c0",
    "text_bright": "#e8e8f4",
}

PLOTLY_LAYOUT_DEFAULTS = dict(
    plot_bgcolor=COLORS["bg"],
    paper_bgcolor=COLORS["bg"],
    font=dict(family="Inter", color=COLORS["text"]),
    margin=dict(l=60, r=30, t=60, b=50),
)


def render_satisfaction_bar_chart(satisfaction_scores, most_disadvantaged_id):

    workers = sorted(satisfaction_scores.keys())
    scores = [satisfaction_scores[w] for w in workers]
    labels = [f"W{w}" for w in workers]


    bar_colors = []
    for w in workers:
        if w == most_disadvantaged_id:
            bar_colors.append(COLORS["rose"])
        else:
            bar_colors.append(COLORS["violet"])

    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=scores,
            marker=dict(
                color=bar_colors,
                line=dict(width=0),
                cornerradius=6,
            ),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Satisfaction Score: <b>%{y}</b>"
                "<extra></extra>"
            ),
        )
    ])

    if scores:
        avg_score = sum(scores) / len(scores)
        fig.add_hline(
            y=avg_score,
            line_dash="dash",
            line_color=COLORS["cyan"],
            line_width=1.5,
            annotation_text=f"Media: {avg_score:.1f}",
            annotation_font=dict(color=COLORS["cyan"], size=11, family="Inter"),
            annotation_position="top right",
        )

    fig.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        title=dict(
            text="Satisfaction Score per Lavoratore",
            font=dict(size=16, color=COLORS["text_bright"], family="Inter"),
            x=0.5, xanchor="center",
        ),
        xaxis=dict(
            title="Lavoratore",
            tickfont=dict(size=10),
            gridcolor=COLORS["grid"],
        ),
        yaxis=dict(
            title="Satisfaction Score",
            gridcolor=COLORS["grid"],
            zeroline=True,
            zerolinecolor=COLORS["grid"],
        ),
        height=400,
        showlegend=False,
        bargap=0.3,
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_fairness_radar_chart(worker_metrics, num_workers):
    workers = sorted(worker_metrics.keys())
    categories = [f"W{w}" for w in workers]
    disadvantaged = [worker_metrics[w]["disadvantaged_shifts"] for w in workers]
    preference = [worker_metrics[w]["preference_score"] for w in workers]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=disadvantaged + [disadvantaged[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(124, 58, 237, 0.15)',
        line=dict(color=COLORS["violet_light"], width=2),
        name='Turni Disagiati',
        hovertemplate="<b>%{theta}</b><br>Turni disagiati: %{r:.1f}<extra></extra>",
    ))

    fig.add_trace(go.Scatterpolar(
        r=preference + [preference[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(6, 182, 212, 0.12)',
        line=dict(color=COLORS["cyan"], width=2),
        name='Preferenze Soddisfatte',
        hovertemplate="<b>%{theta}</b><br>Pref. soddisfatte: %{r}<extra></extra>",
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        title=dict(
            text="Distribuzione Turni Disagiati",
            font=dict(size=16, color=COLORS["text_bright"], family="Inter"),
            x=0.5, xanchor="center",
        ),
        polar=dict(
            bgcolor=COLORS["bg"],
            radialaxis=dict(
                visible=True,
                gridcolor=COLORS["grid"],
                tickfont=dict(size=9, color=COLORS["text"]),
            ),
            angularaxis=dict(
                gridcolor=COLORS["grid"],
                tickfont=dict(size=10, color=COLORS["text"]),
            ),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=11, color=COLORS["text"]),
        ),
        height=450,
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_refinement_line_chart(refinement_history):

    if not refinement_history:
        st.info("Nessuna iterazione di refinement disponibile.")
        return

    iterations = [h["iteration"] for h in refinement_history]
    min_scores = [h["min_score"] for h in refinement_history]
    std_devs = [h.get("std_dev", 0) for h in refinement_history]
    improved = [h.get("improved", True) for h in refinement_history]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=iterations,
        y=min_scores,
        mode='lines+markers',
        name='Min Satisfaction Score',
        line=dict(color=COLORS["violet"], width=3, shape="spline"),
        marker=dict(
            size=10,
            color=[COLORS["emerald"] if imp else COLORS["rose"] for imp in improved],
            line=dict(width=2, color=COLORS["violet_light"]),
            symbol="circle",
        ),
        hovertemplate=(
            "<b>Iterazione %{x}</b><br>"
            "Min Score: <b>%{y}</b><br>"
            "<extra></extra>"
        ),
    ))


    fig.add_trace(go.Scatter(
        x=iterations,
        y=std_devs,
        mode='lines+markers',
        name='Deviazione Standard',
        line=dict(color=COLORS["cyan"], width=2, dash="dot", shape="spline"),
        marker=dict(size=6, color=COLORS["cyan"]),
        yaxis="y2",
        hovertemplate=(
            "<b>Iterazione %{x}</b><br>"
            "Std Dev: <b>%{y:.2f}</b>"
            "<extra></extra>"
        ),
    ))

    layout_settings = PLOTLY_LAYOUT_DEFAULTS.copy()

    specific_settings = dict(
        title=dict(
            text="Andamento Ottimizzazione Fairness",
            font=dict(size=16, color=COLORS["text_bright"], family="Inter"),
            x=1.5, xanchor="center",
        ),
        xaxis=dict(
            title="Iterazione",
            gridcolor=COLORS["grid"],
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            title="Min Satisfaction Score",
            gridcolor=COLORS["grid"],
            zeroline=True,
            zerolinecolor=COLORS["grid"],
        ),
        yaxis2=dict(
            title="Deviazione Standard",
            overlaying="y",
            side="right",
            gridcolor="rgba(0,0,0,0)",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=60, r=40, t=60, b=80),
        height=400,
    )

    layout_settings.update(specific_settings)

    fig.update_layout(**layout_settings)

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_scores_comparison_chart(satisfaction_scores, num_workers):

    sorted_workers = sorted(satisfaction_scores.items(), key=lambda x: x[1])
    labels = [f"Worker {w}" for w, _ in sorted_workers]
    scores = [s for _, s in sorted_workers]

    min_s = min(scores) if scores else 0
    max_s = max(scores) if scores else 1
    range_s = max_s - min_s if max_s != min_s else 1

    bar_colors = []
    for s in scores:
        ratio = (s - min_s) / range_s
        if ratio < 0.33:
            bar_colors.append(COLORS["rose"])
        elif ratio < 0.66:
            bar_colors.append(COLORS["amber"])
        else:
            bar_colors.append(COLORS["emerald"])

    fig = go.Figure(data=[
        go.Bar(
            y=labels,
            x=scores,
            orientation='h',
            marker=dict(
                color=bar_colors,
                cornerradius=4,
            ),
            hovertemplate="<b>%{y}</b><br>Score: %{x}<extra></extra>",
        )
    ])

    fig.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        title=dict(
            text="Classifica Satisfaction Score",
            font=dict(size=16, color=COLORS["text_bright"], family="Inter"),
            x=0.5, xanchor="center",
        ),
        xaxis=dict(
            title="Satisfaction Score",
            gridcolor=COLORS["grid"],
        ),
        yaxis=dict(
            tickfont=dict(size=11),
        ),
        height=max(350, num_workers * 28),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
