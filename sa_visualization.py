"""Plotly-Abbildungen der Simulated-Annealing-Demo: Karten (aus der Hill-Climbing-Demo übernommen), Annahmewahrscheinlichkeit, Abkühlkurven, Verlauf, Sweeps, Heatmap, Streuung.
Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sa_constants as C

TOUR_COLOR = "#4c78a8"
SA_COLOR = "#e45756"
HC_COLOR = "#7f7f7f"
HCR_COLOR = "#f58518"
NEW_COLOR = "#54a24b"


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.1), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _map_layout(fig, height=430):
    fig.update_xaxes(range=[-3, C.AREA + 3], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(range=[-3, C.AREA + 3], showgrid=False, zeroline=False, showticklabels=False)
    return _base(fig, height)


def _line_trace(xy, edges, color, name, dash=None, width=2.5, showlegend=True):
    x, y = [], []
    for a, b in edges:
        x += [xy[a, 0], xy[b, 0], None]
        y += [xy[a, 1], xy[b, 1], None]
    return go.Scatter(x=x, y=y, mode="lines", line=dict(color=color, width=width, dash=dash), name=name, hoverinfo="skip", showlegend=showlegend)


def _tour_edges_list(tour):
    t = np.asarray(tour)
    return list(zip(t.tolist(), np.roll(t, -1).tolist()))


def build_instance(xy):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xy[1:, 0], y=xy[1:, 1], mode="markers", marker=dict(size=7, color=TOUR_COLOR, line=dict(width=1, color="white")), name="Stopps", hovertemplate="Stopp %{customdata}<extra></extra>", customdata=np.arange(1, len(xy))))
    fig.add_trace(go.Scatter(x=[xy[0, 0]], y=[xy[0, 1]], mode="markers", marker=dict(size=14, symbol="star", color="#f58518", line=dict(width=1, color="white")), name="Depot"))
    return _map_layout(fig)


def build_tour(xy, tour, ghost=None):
    """Tour; `ghost` (optional) ist eine zweite Tour, die blass darunter gezeichnet wird (z. B. die beste Tour)."""
    fig = go.Figure()
    if ghost is not None:
        fig.add_trace(_line_trace(xy, _tour_edges_list(ghost), "#c9d6e6", "beste Tour", width=6))
    fig.add_trace(_line_trace(xy, _tour_edges_list(tour), TOUR_COLOR, "aktuelle Tour"))
    fig.add_trace(go.Scatter(x=xy[1:, 0], y=xy[1:, 1], mode="markers", marker=dict(size=6, color="white", line=dict(width=1.5, color=TOUR_COLOR)), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[xy[0, 0]], y=[xy[0, 1]], mode="markers", marker=dict(size=14, symbol="star", color="#f58518", line=dict(width=1, color="white")), name="Depot"))
    return _map_layout(fig)


def build_acceptance(temps_km, deltas_km, labels, deterministic=False):
    """Annahmewahrscheinlichkeit gegen Δ für drei Kontrollwerte (Metropolis: exp(-Δ/T); die drei deterministischen Regeln: Stufenfunktion 1 falls Δ <= Kontrollwert, sonst 0 -
    für Threshold Accepting ist der Kontrollwert die Schwelle selbst, für Great Deluge/LAHC der Abstand zwischen Wasserspiegel/Historienwert und der aktuellen Tour), darunter die Verteilung der Nachbar-Deltas einer guten Tour."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.06)
    d = np.asarray(deltas_km, dtype=float)
    top = max(float(np.percentile(d, 95)) if len(d) else 0.0, 2.0 * max(temps_km))
    xs = np.linspace(0.0, top, 400)
    for T_, lab, col in zip(temps_km, labels, ("#d62728", "#ff9896", "#1f77b4")):
        ys = (xs <= T_).astype(float) if deterministic else np.exp(-xs / T_)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=col, width=2.5, shape="hv" if deterministic else "linear"), name=lab), row=1, col=1)
    fig.add_trace(go.Histogram(x=d[(d >= 0) & (d <= top)], marker_color="#bab0ac", nbinsx=40, name="Nachbarn der Tour"), row=2, col=1)
    fig.update_yaxes(title_text="Annahme" if deterministic else "Annahmewahrscheinlichkeit", row=1, col=1, range=[-0.05, 1.05] if deterministic else None)
    fig.update_yaxes(title_text="Nachbarn", row=2, col=1)
    fig.update_xaxes(title_text="Verlängerung Δ der Tour (km)", row=2, col=1)
    return _base(fig, 430)


def build_cooling(temps_unit, accept_rate, worse_rate, rule="metropolis"):
    """Temperatur/Schwelle/Wasserspiegel (logarithmisch) und Annahmequoten über die Stufen. Bei LAHC gibt es keine Stufen/keinen Plan - nur die Annahmequote."""
    label = {"metropolis": "Temperatur", "threshold": "Schwelle", "great_deluge": "Wasserspiegel (über der Schranke)"}.get(rule, "Kontrollwert")
    if rule == "lahc":
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(1, len(accept_rate) + 1)), y=accept_rate, mode="lines", line=dict(color=TOUR_COLOR, width=2.5), name="alle angenommenen"))
        fig.add_trace(go.Scatter(x=list(range(1, len(worse_rate) + 1)), y=worse_rate, mode="lines", line=dict(color=NEW_COLOR, width=2.5, dash="dash"), name="davon Verschlechterungen"))
        fig.update_yaxes(title_text="Anteil angenommener Vorschläge", tickformat=".0%")
        fig.update_xaxes(title_text="Abschnitt (kein Plan - Listenlänge L bestimmt die Annahme)")
        fig.update_layout(legend=dict(orientation="h", y=-0.25))
        return _base(fig, 340)
    x = list(range(1, len(temps_unit) + 1))
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"{label} (Vielfache der mittleren Kantenlänge)", "Anteil angenommener Vorschläge"), horizontal_spacing=0.12)
    fig.add_trace(go.Scatter(x=x, y=temps_unit, mode="lines", line=dict(color=SA_COLOR, width=2.5), name=label), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=accept_rate, mode="lines", line=dict(color=TOUR_COLOR, width=2.5), name="alle angenommenen"), row=1, col=2)
    fig.add_trace(go.Scatter(x=x, y=worse_rate, mode="lines", line=dict(color=NEW_COLOR, width=2.5, dash="dash"), name="davon Verschlechterungen"), row=1, col=2)
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_xaxes(title_text="Stufe")
    fig.update_yaxes(tickformat=".0%", row=1, col=2)
    fig.update_layout(legend=dict(orientation="h", y=-0.25))
    _base(fig, 340)
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))                # Platz für die Titel der Teilbilder
    return fig


def build_rule_comparison(rows):
    """Abstand zur Schranke der vier Annahmeregeln bei gleichem Budget (Balken)."""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[r["label"] for r in rows], y=[r["gap"] for r in rows], marker_color=[SA_COLOR, "#ff9896", "#72b7b2", "#54a24b"], error_y=dict(type="data", array=[r["gap_sd"] for r in rows])))
    fig.update_yaxes(title_text="Abstand zur Schranke (%)")
    return _base(fig, 340)


def build_trace(trace_iter, trace_length, trace_best, bound, hc_length, hcr_length):
    """Länge der aktuellen und der besten Tour über die Vorschläge; Schranke, ein Hill-Climbing-Abstieg und Hill Climbing mit Neustarts als Linien."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trace_iter, y=trace_length, mode="lines", line=dict(color=TOUR_COLOR, width=1.5), name="aktuelle Tour"))
    fig.add_trace(go.Scatter(x=trace_iter, y=trace_best, mode="lines", line=dict(color=SA_COLOR, width=2.5), name="beste Tour"))
    fig.add_hline(y=bound, line=dict(color=HC_COLOR, dash="dot"), annotation_text="untere Schranke", annotation_position="bottom right")
    fig.add_hline(y=hc_length, line=dict(color=HC_COLOR, dash="dash"), annotation_text="ein Hill-Climbing-Abstieg", annotation_position="top right")
    fig.add_hline(y=hcr_length, line=dict(color=HCR_COLOR, dash="dash"), annotation_text="Hill Climbing mit Neustarts", annotation_position="top right")
    fig.update_xaxes(title_text="Vorschläge (bewertete Nachbarn)", type="log")
    fig.update_yaxes(title_text="Länge (km)", range=[bound * 0.95, max(float(np.percentile(trace_length, 60)), hc_length * 1.15)])
    return _base(fig, 320)


DLB_COLOR = "#54a24b"


def build_budget(rows, dlb_rows=None):
    """Abstand zur Schranke über das Budget: Simulated Annealing gegen einen Hill-Climbing-Abstieg und Hill Climbing mit Neustarts (gleiches Budget, voller Rescan).
    `dlb_rows` (optional, dieselben Budgetwerte wie `rows`): Hill Climbing mit Neustarts über Kandidatenliste + Don't-Look-Bits statt vollem Rescan (nur 2-opt gemessen)."""
    xs = [r["value"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["gap"] for r in rows], mode="lines+markers", line=dict(color=SA_COLOR, width=2.5), name="Simulated Annealing (beste Tour)"))
    fig.add_trace(go.Scatter(x=xs, y=[r["final"] for r in rows], mode="lines+markers", line=dict(color=SA_COLOR, width=1.5, dash="dot"), name="Simulated Annealing (letzte Tour)"))
    fig.add_trace(go.Scatter(x=xs, y=[r["hcr"] for r in rows], mode="lines+markers", line=dict(color=HCR_COLOR, width=2.5), name="Hill Climbing mit Neustarts (voller Rescan)"))
    fig.add_trace(go.Scatter(x=xs, y=[r["hc"] for r in rows], mode="lines", line=dict(color=HC_COLOR, width=1.5, dash="dash"), name="ein Abstieg"))
    if dlb_rows is not None:
        fig.add_trace(go.Scatter(x=[r["value"] for r in dlb_rows], y=[r["gap"] for r in dlb_rows], mode="lines+markers", line=dict(color=DLB_COLOR, width=2.5), name="Hill Climbing mit Neustarts (Kandidatenliste + DLB)"))
    fig.update_xaxes(title_text="Budget (Vorschläge = bewertete Nachbarn)", type="log")
    fig.update_yaxes(title_text="Abstand zur Schranke (%)")
    fig.update_layout(legend=dict(orientation="h", y=-0.3))
    return _base(fig, 380)


def build_sweep(rows, param_label, categorical=False, key_labels=None):
    """Abstand zur Schranke (beste Tour, Streuung als Band bzw. Balken), letzte Tour und ein Hill-Climbing-Abstieg über die Werte eines Reglers."""
    xs = [r["value"] for r in rows]
    if key_labels:
        xs = [key_labels.get(x, x) for x in xs]
    gap = np.array([r["gap"] for r in rows])
    sd = np.array([r["gap_sd"] for r in rows])
    fig = go.Figure()
    if categorical:
        fig.add_trace(go.Bar(x=xs, y=gap, error_y=dict(type="data", array=sd), marker_color=SA_COLOR, name="Simulated Annealing (beste Tour)"))
        fig.add_trace(go.Bar(x=xs, y=[r["hc"] for r in rows], marker_color=HC_COLOR, name="ein Hill-Climbing-Abstieg"))
        fig.update_layout(barmode="group")
    else:
        fig.add_trace(go.Scatter(x=xs, y=gap + sd, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xs, y=np.maximum(gap - sd, 0), mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(228,87,86,0.2)", showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xs, y=gap, mode="lines+markers", line=dict(color=SA_COLOR, width=2.5), name="Simulated Annealing (beste Tour)"))
        fig.add_trace(go.Scatter(x=xs, y=[r["final"] for r in rows], mode="lines+markers", line=dict(color=SA_COLOR, width=1.5, dash="dot"), name="letzte Tour"))
        fig.add_trace(go.Scatter(x=xs, y=[r["hc"] for r in rows], mode="lines", line=dict(color=HC_COLOR, width=1.5, dash="dash"), name="ein Hill-Climbing-Abstieg"))
    fig.update_xaxes(title_text=param_label)
    fig.update_yaxes(title_text="Abstand zur Schranke (%)")
    return _base(fig, 340)


def build_heatmap(table):
    """Abstand zur Schranke über Anfangs- und Endtemperatur (leere Zellen: Endtemperatur über der Anfangstemperatur)."""
    z = [[np.nan if v is None else v for v in row] for row in table["gap"]]
    text = [["" if v is None else f"{v:.1f}" for v in row] for row in table["gap"]]
    fig = go.Figure(go.Heatmap(z=z, x=[str(t) for t in table["t_end"]], y=[str(t) for t in table["t0"]], text=text, texttemplate="%{text}", colorscale="RdYlGn_r", zmin=0, zmax=15, colorbar=dict(title="Abstand (%)")))
    fig.update_xaxes(title_text="Endtemperatur (Vielfache der mittleren Kantenlänge)", type="category")
    fig.update_yaxes(title_text="Anfangstemperatur", type="category")
    return _base(fig, 340)


def build_spread(sa, hc):
    """Verteilung des Abstands zur Schranke über viele Ketten-Seeds derselben Instanz: Simulated Annealing gegen einen Hill-Climbing-Abstieg aus derselben Startlösung."""
    top = max(float(np.max(hc)), float(np.max(sa))) + 1
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=hc, xbins=dict(start=0, end=top, size=1.0), marker_color=HC_COLOR, opacity=0.7, name="ein Hill-Climbing-Abstieg"))
    fig.add_trace(go.Histogram(x=sa, xbins=dict(start=0, end=top, size=1.0), marker_color=SA_COLOR, opacity=0.7, name="Simulated Annealing (beste Tour)"))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title_text="Abstand zur Schranke (%)")
    fig.update_yaxes(title_text="Anzahl Ketten")
    return _base(fig, 300)


def build_scaling(blocks):
    """Abstand zur Schranke über die Stoppzahl für die Budget-Regeln in `blocks` ({label, rows}), dazu ein Hill-Climbing-Abstieg."""
    fig = go.Figure()
    colors = [SA_COLOR, "#f58518"]
    for k, blk in enumerate(blocks):
        fig.add_trace(go.Scatter(x=[r["value"] for r in blk["rows"]], y=[r["gap"] for r in blk["rows"]], mode="lines+markers", line=dict(color=colors[k % 2], width=2.5), name=f"Simulated Annealing, {blk['label']}"))
    fig.add_trace(go.Scatter(x=[r["value"] for r in blocks[0]["rows"]], y=[r["hc"] for r in blocks[0]["rows"]], mode="lines", line=dict(color=HC_COLOR, dash="dash"), name="ein Hill-Climbing-Abstieg"))
    fig.update_xaxes(title_text="Stopps")
    fig.update_yaxes(title_text="Abstand zur Schranke (%)")
    return _base(fig, 340)
