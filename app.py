"""Simulated Annealing - eine Lieferrunde, die den ersten Hügel überwindet - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - Simulated Annealing auf derselben Lieferrunde wie die Hill-Climbing-Demo -
und lässt stattdessen das Beispiel wachsen. Zweites Stück der Trajektorien-Metaheuristiken-Linie der "Konzepte"-Reihe: dieselbe Suche wie die Wurzel, aber sie nimmt Verschlechterungen mit
wachsender Zurückhaltung an. Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time
from dataclasses import replace

import numpy as np
import streamlit as st

import sa_constants as C
import sa_tour as T
from sa_evaluation import SWEEP_LABELS, Settings, analyse, chain_spread, dlb_budget_sweep, heatmap_table, rule_comparison_table, scaling_table, sweep, verdict
from sa_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_chain_seed,
    randomize_seed,
    sync_query_params,
)
from sa_visualization import build_acceptance, build_budget, build_cooling, build_heatmap, build_instance, build_rule_comparison, build_scaling, build_spread, build_sweep, build_tour, build_trace

st.set_page_config(page_title="Simulated Annealing – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _sweep(param, base):
    return sweep(param, base)


@st.cache_data(show_spinner=False)
def _dlb_budget(base):
    return dlb_budget_sweep(base=base)


@st.cache_data(show_spinner=False)
def _heatmap(base):
    return heatmap_table(base)


@st.cache_data(show_spinner=False)
def _spread(base):
    return chain_spread(base)


@st.cache_data(show_spinner=False)
def _scaling(base):
    return scaling_table(base)


@st.cache_data(show_spinner=False)
def _rule_comparison(base, budget):
    return rule_comparison_table(base, budget=budget)


def _fmt_int(x):
    return f"{int(round(x)):,}".replace(",", ".")


st.title("🌡️ Simulated Annealing – eine Lieferrunde, die Hügel überwindet")
st.markdown(
    """
Das **Hill Climbing** der vorigen Demo nimmt nur Züge an, die die Tour kürzer machen – und bleibt im ersten lokalen Optimum stecken. **Simulated Annealing** (simuliertes Abkühlen) ändert genau **eine** Regel: ein Zug, der die Tour um $\\Delta$ km **verlängert**,
wird trotzdem mit der Wahrscheinlichkeit $e^{-\\Delta/T}$ angenommen. Die **Temperatur** $T$ ist anfangs hoch (fast alles wird angenommen, die Suche wandert frei) und sinkt nach einem **Abkühlplan** (zum Schluss wird fast nur noch bergab gegangen).
Ob das den Hügeln entkommt, was es kostet und wovon es abhängt – Temperaturbereich, Budget, Abkühlplan, Nachbarschaft –, misst diese Demo, gegen dieselbe untere Schranke und dasselbe Bewertungsbudget wie das Hill Climbing, mit Siegen und Niederlagen.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo – zweites Stück der Trajektorien-Metaheuristiken-Linie der \"Konzepte\"-Reihe – **ein** Verfahren an einem wachsenden Beispiel. "
    "Die Linie hat keinen Konvergenzpunkt: Simulated Annealing ist **eine** Antwort auf die Schwäche der Wurzel (Hill Climbing); Iterated Local Search mit VNS und ALNS, Tabu Search und GRASP sind andere. "
    "Vehikel ist dieselbe Rundtour wie in der Hill-Climbing-Demo: ein Depot in der Mitte, n Kundenstopps in einem 100 × 100-km-Gebiet, euklidische Entfernungen."
)

with st.expander("So funktioniert Simulated Annealing", expanded=True):
    st.markdown(
        """
1. **Ein Vorschlag = ein bewerteter Nachbar.** In jeder Iteration wird **ein** zufälliger Nachbar der Tour gezogen (2-opt: ein Stück umdrehen, Or-opt: ein Stück versetzen, Tausch, oder eine Mischung) und seine Längenänderung $\\Delta$ aus wenigen Kanten berechnet.
   Das Hill Climbing bewertet dagegen *alle* Nachbarn und nimmt einen; beide zählen ihre Bewertungen, das **Budget** ist also vergleichbar.
2. **Annahmeregel.** Vier zur Wahl. **Metropolis** (Voreinstellung): ist $\\Delta \\le 0$, wird der Zug angenommen, sonst mit der Wahrscheinlichkeit $e^{-\\Delta/T}$ – bei $T$ gleich $\\Delta$ etwa 37 %, bei $\\Delta = 3T$ etwa 5 %. Die anderen drei entscheiden **deterministisch**, ohne Zufallszahl:
   **Threshold Accepting** nimmt an, wenn $\\Delta$ höchstens eine Schwelle ist (dieselbe Skala wie Metropolis' Temperatur); **Great Deluge** vergleicht die **absolute** neue Tourlänge mit einem sinkenden Wasserspiegel (nicht die Änderung); **Late Acceptance Hill Climbing** vergleicht mit der Tourlänge von vor $L$ Vorschlägen ODER der aktuellen Tour – kein Plan, nur ein Regler ($L$).
   Bei 200 Tausend Vorschlägen: Metropolis 1.4 %, Late Acceptance 2.2 %, Threshold Accepting 2.7 %, Great Deluge 3.2 % über der Schranke – der Zufall bringt gegenüber den drei deterministischen Regeln noch einen kleinen Vorsprung.
3. **Abkühlplan.** Temperatur/Schwelle/Wasserspiegel fallen in Stufen (je Stufe bleibt der Wert gleich): **geometrisch** (jede Stufe um denselben Faktor kleiner), **linear**, oder **logarithmisch** ($T_0 \\ln 2 / \\ln(k+2)$, nach Hajek in unendlich langer Zeit optimal). Late Acceptance braucht keinen Plan.
   Angegeben wird der Wert in Vielfachen der **mittleren Kantenlänge einer guten Tour** (untere Schranke geteilt durch die Zahl der Knoten, bei 60 Stopps etwa 10 km) – so passt derselbe Regler zu jeder Instanz (außer bei Great Deluge, dessen Anfangsabstand die GESAMTE Tourlänge über der Schranke decken muss, nicht nur einen Zug – siehe Seitenleiste).
4. **Gemerkt wird die beste Tour.** Die Kette wandert, auch bergauf; die kürzeste je besuchte Tour wird aufgehoben. Am Ende zählt sie, nicht die letzte.
5. **Bewertung.** Der Abstand zur **1-Baum-Schranke** (Held-Karp), wie in der Hill-Climbing-Demo: bei gleichverteilten 60 Stopps liegt sie im Mittel 0.5 % unter dem echten Optimum. Verglichen wird mit **einem Abstieg** und mit **Hill Climbing mit Neustarts** bei gleichem Bewertungsbudget.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_names = list(C.PRESETS.keys())
for row in (preset_names[:4], preset_names[4:]):
    cols = st.columns(len(row))
    for col, name in zip(cols, row):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name], key=f"preset_{name}")

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_stops = st.slider(
        "Stopps", *bounds("n_slider"), key="n_slider", step=C.N_STEP,
        help="Anzahl der Kundenstopps (das Depot kommt dazu). Bei 200 Tausend Vorschlägen, 2-opt und geometrischem Plan von 0.5 auf 0.1 endet die beste Tour bei 10 / 20 / 40 / 60 / 100 / 150 / 200 Stopps 0.0 / 0.05 / 0.9 / 1.4 / 3.8 / 6.3 / 7.7 % über der Schranke "
             "(ein Hill-Climbing-Abstieg: 1.7 / 1.9 / 7.7 / 7.9 / 9.1 / 9.1 / 9.5 %). Mit 5 000 Vorschlägen je Stopp: 0.05 / 0.9 / 1.4 / 2.0 / 3.8 / 4.1 % bei 20 / 40 / 60 / 100 / 150 / 200 Stopps.",
    )
    cluster_share = st.slider(
        "Anteil der Stopps in Gruppen [%]", *bounds("ballung_slider"), key="ballung_slider", step=C.BALLUNG_STEP,
        help="Wie viele Stopps in fünf Gruppen (Städten) liegen statt gleichverteilt im Gebiet. Bei 0 / 25 / 50 / 75 / 100 % liegt die beste Tour 1.4 / 1.7 / 1.5 / 1.1 / 1.9 % über der Schranke (60 Stopps, Standardplan); "
             "ein Hill-Climbing-Abstieg endet bei 7.9 / 8.0 / 5.7 / 4.1 / 4.4 %.",
    )
    neighborhood = st.selectbox(
        "Nachbarschaft", list(C.NEIGHBORHOOD_LABELS), key="neighborhood_select", format_func=lambda k: C.NEIGHBORHOOD_LABELS[k],
        help="Welche Änderung ein Vorschlag ist. Tausch 22.6 %, 2-opt 1.4 %, Or-opt 5.7 %, 2-opt + Or-opt 1.2 % über der Schranke (60 Stopps, Standardplan); ein Hill-Climbing-Abstieg mit derselben Nachbarschaft: 54.5 / 7.9 / 10.4 / 3.7 %. "
             "Die Temperatur ersetzt die Nachbarschaft nicht (Tausch allein bleibt schlecht), aber 2-opt allein kommt weit: 2-opt + Or-opt ändert kaum etwas (1.2 statt 1.4 %), beim Hill Climbing halbiert es die Lücke.",
    )
    rule = st.selectbox(
        "Annahmeregel", list(C.RULE_LABELS), key="rule_select", format_func=lambda k: C.RULE_LABELS[k],
        help="Wie ein Vorschlag angenommen wird, der die Tour verlängert. **Metropolis** (Zufall, Wahrscheinlichkeit $e^{-\\Delta/T}$) ist die Voreinstellung; **Threshold Accepting**, **Great Deluge** und **Late Acceptance Hill Climbing** entscheiden deterministisch, ohne Zufallszahl. "
             "Bei 200 Tausend Vorschlägen (60 Stopps, 2-opt, Standardplan): Metropolis 1.4 %, Late Acceptance 2.2 %, Threshold Accepting 2.7 %, Great Deluge 3.2 % über der Schranke – Hill Climbing mit Neustarts bei gleichem Budget 4.9 %. "
             "Alle vier schlagen den Neustart-Abstieg klar; der Zufall bringt gegenüber den drei deterministischen Regeln noch einen kleinen Vorsprung.",
    )
    if rule in ("metropolis", "threshold"):
        schedule = st.selectbox(
            "Abkühlplan", list(C.SCHEDULE_LABELS), key="schedule_select", format_func=lambda k: C.SCHEDULE_LABELS[k],
            help="Wie die Temperatur (Metropolis) bzw. die Schwelle (Threshold Accepting) von T0 auf T_end fällt. Bei 200 Tausend Vorschlägen (60 Stopps, 2-opt, Metropolis) liegt die beste Tour geometrisch 1.4 %, linear 1.3 % und logarithmisch 3.4 % über der Schranke: geometrisch und linear sind gleichwertig, "
                 "der logarithmische Plan (in der Theorie optimal, erreicht T_end nicht) kühlt zu langsam ab.",
        )
        st.session_state["_kept_schedule_select"] = schedule
    else:
        schedule = st.session_state.get("_kept_schedule_select", C.DEFAULT_SCHEDULE)
        st.caption(("Great Deluge nutzt den Abkühlplan für die Form des Wasserspiegels (siehe Anfangs-/Endabstand unten)." if rule == "great_deluge"
                    else "Late Acceptance Hill Climbing braucht keinen Plan – nur die Listenlänge L unten."))
    t0_label = "Anfangstemperatur T0" if rule == "metropolis" else "Anfangsschwelle" if rule == "threshold" else "T0"
    if rule in ("metropolis", "threshold"):
        t0 = st.slider(
            t0_label, *bounds("t0_slider"), key="t0_slider", step=C.T0_STEP, format="%.2f",
            help=("In Vielfachen der mittleren Kantenlänge einer guten Tour. Bei T0 = 0.1 / 0.25 / 0.5 / 1 / 2 / 4 (T_end 0.1, 200 Tausend Vorschläge, 60 Stopps, Metropolis) liegt die beste Tour 3.8 / 2.1 / 1.4 / 1.2 / 1.7 / 1.8 % über der Schranke; "
                  "der Anteil angenommener Vorschläge ist 0.3 / 0.6 / 1.2 / 3.0 / 7.6 / 15.2 %. Der Anfang ist unkritisch (0.25 bis 4 geht), zu kalt (0.1) verschenkt viel." if rule == "metropolis" else
                  "Dieselbe Skala wie Metropolis' Temperatur, hier als feste Schwelle für die Längenänderung eines Zuges: ein Vorschlag wird angenommen, wenn Δ höchstens die Schwelle ist – ohne Zufall."),
        )
        st.session_state["_kept_t0_slider"] = t0
        if schedule != "log":
            t_end = st.slider(
                "Endtemperatur" if rule == "metropolis" else "Endschwelle", *bounds("tend_slider"), key="tend_slider", step=C.T_END_STEP, format="%.3f",
                help=("In Vielfachen der mittleren Kantenlänge einer guten Tour (T0 = 0.5). Bei 0.005 / 0.02 / 0.05 / 0.1 / 0.2 / 0.5 (Metropolis) liegt die beste Tour 3.1 / 2.3 / 1.75 / 1.4 / 1.4 / 9.5 % über der Schranke, die letzte 3.1 / 2.4 / 2.0 / 1.8 / 3.7 / 27.2 %: "
                      "das Ende ist kritisch – zu heiß (0.5) kommt die Kette nicht zur Ruhe, zu kalt (0.005) verschwendet Vorschläge." if rule == "metropolis" else
                      "Schwelle der letzten Stufe (Threshold Accepting): zu hoch, und die Suche kommt nicht zur Ruhe, zu niedrig verschwendet Vorschläge – wie bei Metropolis' Endtemperatur."),
            )
            st.session_state["_kept_tend_slider"] = t_end
        else:
            t_end = float(st.session_state.get("_kept_tend_slider", C.DEFAULT_T_END))
            st.caption("Der logarithmische Plan hat keine Endtemperatur/-schwelle: er endet bei $T_0 \\ln 2 / \\ln(K + 1)$.")
    else:
        t0 = float(st.session_state.get("_kept_t0_slider", C.DEFAULT_T0))
        t_end = float(st.session_state.get("_kept_tend_slider", C.DEFAULT_T_END))
    if rule == "great_deluge":
        gd_t0 = st.slider(
            "Anfangsabstand", C.GD_T0_MIN, C.GD_T0_MAX, float(st.session_state.get("gd_t0_slider", C.DEFAULT_GD_T0)), key="gd_t0_slider", step=C.GD_T0_STEP,
            help="Wasserspiegel zu Beginn: Vielfache der mittleren Kantenlänge ÜBER der Schranke (absolute Tourlänge, nicht eine Änderung pro Zug). Bei einer zufälligen Startlösung liegt die Tour selbst schon 250 bis 280 Einheiten über der Schranke; "
                 "ist der Anfangsabstand knapp darunter (280), kippt die Suche gelegentlich katastrophal (die Demo hebt den Spiegel dann automatisch auf die Startlänge an, damit das nicht passiert – aber die Kalibrierung gilt für 60 Stopps, nicht für viel größere Instanzen).",
        )
        gd_t_end = st.slider(
            "Endabstand", C.GD_T_END_MIN, C.GD_T_END_MAX, float(st.session_state.get("gd_tend_slider", C.DEFAULT_GD_T_END)), key="gd_tend_slider", step=C.GD_T_END_STEP,
            help="Wasserspiegel am Ende, Einheiten über der Schranke. Bei t0=300: Endabstand 2 ergibt 3.4 %, deutlich kleiner (unter 295) kippt die Suche katastrophal (Streuung größer als der Mittelwert), größer (6) verschwendet Vorschläge (7.3 %).",
        )
    else:
        gd_t0 = float(st.session_state.get("gd_t0_slider", C.DEFAULT_GD_T0))
        gd_t_end = float(st.session_state.get("gd_tend_slider", C.DEFAULT_GD_T_END))
    budget = st.select_slider(
        "Budget (Vorschläge)", options=list(C.BUDGETS), key="budget_select", format_func=_fmt_int,
        help="Wie viele Nachbarn insgesamt bewertet werden (bei 60 Stopps braucht ein Hill-Climbing-Abstieg etwa 74 Tausend). Bei 10 / 25 / 50 / 100 / 200 Tausend / 0.5 / 1 / 2 Millionen liegt die beste Tour (Metropolis) 8.0 / 4.1 / 2.6 / 2.6 / 1.4 / 1.0 / 0.7 / 0.5 % über der Schranke, "
             "die letzte 8.0 / 4.5 / 3.3 / 3.1 / 1.8 / 1.8 / 1.25 / 1.2 %; Hill Climbing mit Neustarts: 7.9 / 7.9 / 7.9 / 7.9 / 4.9 / 2.9 / 2.5 / 1.9 %. Bei 10 Tausend ist Simulated Annealing nicht besser als ein Abstieg.",
    )
    if rule == "lahc":
        lahc_length = st.slider(
            "Listenlänge L", C.LAHC_MIN, C.LAHC_MAX, int(st.session_state.get("lahc_length_slider", C.DEFAULT_LAHC_L)), key="lahc_length_slider", step=C.LAHC_STEP,
            help="Länge des Ringpuffers vergangener Tourlängen (in Vorschlägen), gegen den ein Kandidat zusätzlich zur aktuellen Tour antritt. Bei 200 Tausend Vorschlägen (60 Stopps): L=100 4.0 %, L=300 3.9 %, L=500 2.2 % (Voreinstellung) – "
                 "darüber kippt es: L=700 15.4 %, L=1000 44.2 %, L=3000 147 %. Der Ringpuffer 'kommt nicht mehr nach': bei zu großem L vergleicht ein Vorschlag noch lange mit einem sehr alten, schlechten Wert und wird fast immer angenommen – die Suche konvergiert nicht mehr.",
        )
    else:
        lahc_length = int(st.session_state.get("lahc_length_slider", C.DEFAULT_LAHC_L))
    if rule in ("metropolis", "threshold", "great_deluge"):
        levels = st.select_slider(
            "Temperaturstufen", options=list(C.LEVEL_OPTIONS), key="levels_select",
            help="In wie viele Stufen gleicher Temperatur/Schwelle/Wasserspiegel das Budget zerlegt wird. Bei 1 / 3 / 10 / 30 / 100 / 300 / 1000 Stufen (Metropolis) liegt die beste Tour 8.9 / 2.2 / 1.1 / 1.7 / 1.4 / 1.3 / 1.6 % über der Schranke: eine Stufe (letzte Tour 28.4 %) ist Unsinn, "
                 "ab etwa drei Stufen ändert sich nichts Sicheres mehr.",
        )
        st.session_state["_kept_levels_select"] = levels
    else:
        levels = int(st.session_state.get("_kept_levels_select", C.DEFAULT_LEVELS))
        st.caption("Late Acceptance Hill Climbing kennt keine Temperaturstufen – nur den einen Regler L.")
    start = st.radio(
        "Startlösung", list(C.START_LABELS), key="start_radio", format_func=lambda k: C.START_LABELS[k], horizontal=True,
        help="Zufällige Reihenfolge oder Nächster Nachbar. Die erste Temperatur ist heiß genug, um die Startlösung zu vergessen: zufällig 1.4 %, Nächster Nachbar 2.0 % über der Schranke (60 Stopps, Standardplan) – kein Gewinn. "
             "Ein Hill-Climbing-Abstieg endet bei 7.9 % (zufällig) und 7.3 % (Nächster Nachbar).",
    )
    seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Seed für die Lage der Stopps.")
    chain_seed = st.number_input(
        "Zufalls-Seed der Kette", *bounds("chain_seed_input"), key="chain_seed_input", step=1,
        help="Steuert die zufälligen Vorschläge und Annahmen (und die zufällige Startlösung). Bei 20 Ketten auf der Standardinstanz liegt die beste Tour 0.1 bis 5.1 % über der Schranke (Standardabweichung 1.0 Prozentpunkte), "
             "85 % der Ketten enden höchstens 2 % darüber; ein Hill-Climbing-Abstieg streut von 0.8 bis 11.5 % (Standardabweichung 2.4), nur 5 % enden höchstens 2 % darüber.",
    )
    st.button("🎲 Neue Kette würfeln", width="stretch", on_click=randomize_chain_seed, help="Würfelt einen neuen Seed für dieselbe Instanz.")

sync_query_params({
    "n_slider": int(n_stops), "ballung_slider": int(cluster_share), "seed_input": int(seed), "neighborhood_select": neighborhood, "rule_select": rule, "schedule_select": schedule, "t0_slider": round(float(t0), 4),
    "tend_slider": round(float(t_end), 4), "budget_select": int(budget), "levels_select": int(levels), "start_radio": start, "chain_seed_input": int(chain_seed),
    "lahc_length_slider": int(lahc_length), "gd_t0_slider": round(float(gd_t0), 4), "gd_tend_slider": round(float(gd_t_end), 4),
})

t_end_used = min(float(t_end), float(t0))
if rule in ("metropolis", "threshold") and schedule != "log" and float(t_end) > float(t0):
    st.sidebar.warning("Die Endtemperatur liegt über der Anfangstemperatur – sie wird auf T0 begrenzt (konstante Temperatur).")
gd_t_end_used = min(float(gd_t_end), float(gd_t0))
if rule == "great_deluge" and float(gd_t_end) > float(gd_t0):
    st.sidebar.warning("Der Endabstand liegt über dem Anfangsabstand – er wird auf den Anfangsabstand begrenzt (konstanter Spiegel).")
settings = Settings(int(n_stops), int(cluster_share), int(seed), neighborhood, schedule, round(float(t0), 4), round(t_end_used, 4), int(budget), int(levels), start, int(chain_seed),
                     rule, int(lahc_length), round(float(gd_t0), 4), round(gd_t_end_used, 4))
with st.spinner("Rechne..."):
    a = _analysis(settings)
run = a.run
xy = a.inst.xy
code = verdict(a)
data_key = settings
n_levels = len(run.temps)
temps_unit = run.temps / a.unit

# --- Simulated Annealing in Aktion ---------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Simulated Annealing in Aktion")
STEP_LABELS = {1: "1 · Instanz", 2: "2 · Temperatur", 3: "3 · Abkühlen", 4: "4 · Tour bei sinkender Temperatur", 5: "5 · Ergebnis"}
if "sa_step" not in st.session_state or st.session_state.get("sa_step_owner") != data_key:
    st.session_state["sa_step"] = 1
    st.session_state["sa_step_owner"] = data_key
    st.session_state.pop("sa_level", None)
step_col, play_col = st.columns([5, 2])
with step_col:
    step = st.select_slider("Schritt", options=list(STEP_LABELS), key="sa_step", format_func=lambda s: STEP_LABELS[s])
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")

level = n_levels
play_levels = False
if step == 4 and n_levels > 1:
    lv_col, lvplay_col = st.columns([5, 2])
    with lv_col:
        level = st.slider("Temperaturstufe", 1, n_levels, value=n_levels, key="sa_level", help="Die Tour am Ende dieser Temperaturstufe (1 = heiß, letzte = kalt).")
    with lvplay_col:
        play_levels = st.button("▶️ Abkühlen abspielen", width="stretch")
view_slot = st.empty()


def _frames():
    if n_levels == 1:
        return [1]
    return sorted({int(round(x)) for x in np.linspace(1, n_levels, min(n_levels, 40))})


def _render(current_step, lv):
    with view_slot.container():
        if current_step == 1:
            st.markdown(f"**{a.inst.n} Kundenstopps und das Depot (Stern)** – {a.inst.cluster_share} % der Stopps in Gruppen")
            st.plotly_chart(build_instance(xy), width="stretch", key="s1_map")
        elif current_step == 2:
            c1, c2 = st.columns([3, 2])
            deltas = T.neighbor_deltas(run.best_tour, a.D, settings.neighborhood)
            if settings.rule == "lahc":
                c1.markdown("**Verlängerungen der Nachbarn der besten Tour** – Late Acceptance hat keine Temperatur/Schwelle: ein Vorschlag wird angenommen, wenn er kürzer ist als die Tour vor L Vorschlägen ODER kürzer als die aktuelle Tour.")
                c1.plotly_chart(build_cooling(temps_unit, run.accept_rate, run.worse_rate, rule="lahc"), width="stretch", key="s2_lahc")
                c2.markdown("**Was L bedeutet**")
                c2.metric("Listenlänge L", f"{settings.lahc_length:,}".replace(",", "."), help="Länge des Ringpuffers vergangener Tourlängen (in Vorschlägen).")
            else:
                is_det = settings.rule in ("threshold", "great_deluge")
                if settings.rule == "great_deluge":
                    lvl0 = a.bound + run.temps[0] - run.level_length[0]
                    lvl_mid = a.bound + run.temps[len(run.temps) // 2] - run.level_length[len(run.temps) // 2]
                    lvl_end = a.bound + run.temps[-1] - run.level_length[-1]
                    ctrl = [max(0.0, lvl0), max(0.0, lvl_mid), max(0.0, lvl_end)]
                    ctrl_labels = [f"Anfang: Spiegel − Tour ≈ {ctrl[0]:.1f} km", f"Mitte: ≈ {ctrl[1]:.1f} km", f"Ende: ≈ {ctrl[2]:.1f} km"]
                    title = "**Annahme einer Verlängerung um Δ km** (Wasserspiegel abzüglich der aktuellen Tour an drei Stufen) – und wie lang die Verlängerungen bei einer guten Tour tatsächlich sind"
                else:
                    t_mid = float(np.sqrt(run.temps[0] * run.temps[-1])) if settings.rule == "metropolis" else float((run.temps[0] + run.temps[-1]) / 2)
                    ctrl = [float(run.temps[0]), t_mid, float(run.temps[-1])]
                    unit_word = "T" if settings.rule == "metropolis" else "Schwelle"
                    ctrl_labels = [f"Anfang: {unit_word} = {ctrl[0]:.1f} km", f"Mitte: {unit_word} = {ctrl[1]:.1f} km", f"Ende: {unit_word} = {ctrl[2]:.2f} km"]
                    title = "**Annahmewahrscheinlichkeit einer Verlängerung um Δ km**" if settings.rule == "metropolis" else "**Annahme einer Verlängerung um Δ km** (Threshold Accepting: deterministisch)"
                    title += " – und wie lang die Verlängerungen bei einer guten Tour tatsächlich sind"
                c1.markdown(title)
                c1.plotly_chart(build_acceptance(ctrl, deltas[deltas > 0], ctrl_labels, deterministic=is_det), width="stretch", key="s2_acc")
                c2.markdown(f"**Was {'die Temperatur' if settings.rule == 'metropolis' else 'die Schwelle' if settings.rule == 'threshold' else 'der Wasserspiegel'} bedeutet**")
                c2.metric("Temperatureinheit", f"{a.unit:.1f} km", help="Mittlere Kantenlänge einer guten Tour: untere Schranke geteilt durch die Zahl der Knoten.")
                c2.metric("Anfang → Ende", f"{ctrl[0]:.1f} → {ctrl[-1]:.2f} km", help="Kontrollwert der ersten und der letzten Stufe in km.")
            improving = int((deltas < -1e-9).sum())
            c2.metric("Nachbarn der besten Tour", f"{len(deltas):,}".replace(",", "."), delta=f"{improving} verkürzen", delta_color="off", help="Zahl der Nachbarn der besten gefundenen Tour und wie viele davon sie noch verkürzen.")
        elif current_step == 3:
            st.markdown("**Kontrollwert und Annahmequote je Stufe**" if settings.rule != "lahc" else "**Annahmequote über den Lauf**")
            st.plotly_chart(build_cooling(temps_unit, run.accept_rate, run.worse_rate, rule=settings.rule), width="stretch", key="s3_cooling")
            st.markdown("**Länge der aktuellen und der besten Tour über die Vorschläge**")
            st.plotly_chart(build_trace(run.trace_iter, run.trace_length, run.trace_best, a.bound, a.hc.length, T.tour_length(a.hcr_tour, a.D)), width="stretch", key="s3_trace")
        elif current_step == 4:
            c1, c2 = st.columns([3, 2])
            snap = run.snapshots[lv - 1]
            stage_word = "Abschnitt" if settings.rule == "lahc" else "Stufe"
            ctrl_word = "" if settings.rule == "lahc" else f": {'T' if settings.rule == 'metropolis' else 'Schwelle' if settings.rule == 'threshold' else 'Spiegel-Abstand'} = {run.temps[lv - 1]:.2f} km"
            c1.markdown(f"**{stage_word} {lv} von {n_levels}{ctrl_word}** – aktuelle Tour {T.tour_length(snap, a.D):,.0f} km".replace(",", "."))
            c1.plotly_chart(build_tour(xy, snap, ghost=run.best_tour if lv < n_levels else None), width="stretch", key=f"s4_map_{lv}")
            c2.markdown("**An dieser Stufe**")
            c2.metric("Temperatur", f"{temps_unit[lv - 1]:.2f} Einheiten", help="Vielfache der mittleren Kantenlänge einer guten Tour.")
            c2.metric("Angenommene Vorschläge", f"{run.accept_rate[lv - 1]:.1%}", delta=f"davon Verschlechterungen {run.worse_rate[lv - 1]:.1%}", delta_color="off", help="Anteil der Vorschläge dieser Stufe, die angenommen wurden.")
            c2.metric("Beste Tour bis hier", f"{run.level_best[lv - 1]:,.0f} km".replace(",", "."), delta=f"aktuell {run.level_length[lv - 1]:,.0f} km".replace(",", "."), delta_color="off",
                      help="Länge der kürzesten bisher besuchten Tour; im Delta die der aktuellen.")
        else:
            c1, c2 = st.columns(2)
            c1.markdown(f"**Simulated Annealing: beste Tour** – {a.gap:.1f} % über der Schranke")
            c1.plotly_chart(build_tour(xy, run.best_tour), width="stretch", key="s5_sa")
            c2.markdown(f"**Hill Climbing mit Neustarts** ({a.hcr_starts} Abstieg{'e' if a.hcr_starts != 1 else ''}, gleiches Budget) – {a.hcr_gap:.1f} % über der Schranke")
            c2.plotly_chart(build_tour(xy, a.hcr_tour), width="stretch", key="s5_hc")


if auto_play:
    for s in STEP_LABELS:
        if s == 4:
            for f in _frames():
                _render(4, f)
                time.sleep(0.1)
            time.sleep(0.6)
        else:
            _render(s, n_levels)
            time.sleep(1.2)
    step = 5
elif play_levels:
    for f in _frames():                                                    # das letzte Bild ist die letzte Stufe
        _render(4, f)
        time.sleep(0.1)
else:
    _render(step, level)

if step == 1:
    st.caption(f"{a.inst.n} Stopps; die untere Schranke der kürzesten Rundtour liegt bei {a.bound:,.0f} km (1-Baum-Schranke, Held-Karp). Die Temperatureinheit ist {a.unit:.1f} km.".replace(",", "."))
elif step == 2:
    if settings.rule == "metropolis":
        st.caption(f"Oben: bei der Anfangstemperatur ({run.temps[0]:.1f} km) wird eine Verlängerung um 5 km mit {np.exp(-5 / run.temps[0]):.0%}, bei der Endtemperatur ({run.temps[-1]:.2f} km) mit {np.exp(-5 / run.temps[-1]):.1%} angenommen. "
                   "Unten: die Verlängerungen aller Nachbarn der besten gefundenen Tour – bei kleinem T sind nur die kleinsten davon noch erreichbar.")
    elif settings.rule == "threshold":
        st.caption(f"Oben: bei der Anfangsschwelle ({run.temps[0]:.1f} km) wird jede Verlängerung bis 5 km angenommen, bei der Endschwelle ({run.temps[-1]:.2f} km) nur noch, wenn sie darunterbleibt – deterministisch, keine Zufallszahl. "
                   "Unten: die Verlängerungen aller Nachbarn der besten gefundenen Tour.")
    elif settings.rule == "great_deluge":
        st.caption("Oben: eine Verlängerung wird angenommen, wenn die neue Tourlänge unter dem Wasserspiegel bleibt – gezeigt als Abstand Spiegel minus aktuelle Tour an drei Stufen, deterministisch. "
                   "Unten: die Verlängerungen aller Nachbarn der besten gefundenen Tour.")
    else:
        st.caption(f"Kein Plan, keine Temperatur: bei L = {settings.lahc_length:,} Vorschlägen vergleicht Late Acceptance jeden Kandidaten mit der Tour von vor L Vorschlägen UND mit der aktuellen Tour – kürzer als eine der beiden reicht.".replace(",", "."))
elif step == 3:
    stage_noun = "Abschnitten" if settings.rule == "lahc" else "Stufen"
    st.caption(f"{_fmt_int(run.proposals)} Vorschläge in {n_levels} {stage_noun}; angenommen wurden {run.accepted / run.proposals:.1%}, davon {a.worse_share:.0%} Verschlechterungen. Die aktuelle Tour (blau) schwankt zuerst stark und beruhigt sich mit sinkendem Kontrollwert; die beste Tour (rot) fällt nur.")
elif step == 4:
    st.caption("Die aktuelle Tour bei der gewählten Stufe (blau), darunter blass die beste Tour am Ende. Bei hoher Temperatur ist die Tour ein Wirrwarr, bei sinkender bildet sich die Struktur heraus.")
else:
    st.caption(f"Links die beste Tour der Kette, rechts die beste Tour aus {a.hcr_starts} Hill-Climbing-Abstiegen mit demselben Bewertungsbudget ({_fmt_int(settings.budget)}).")

st.markdown("---")

# --- Ergebnis -------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was die Kette gefunden hat")
st.caption(
    "**Abstand zur Schranke:** Länge der Tour gegenüber einer unteren Schranke der kürzesten Rundtour (1-Baum, Held-Karp) in Prozent. Das wahre Optimum liegt bei gleichverteilten 60 Stopps im Mittel 0.5 % über der Schranke, der Abstand überschätzt die echte Lücke um diesen Betrag. "
    "Ein Lauf ist eine Ziehung: die Ketten streuen (siehe Streuung unten), Vergleiche gelten für diesen Lauf."
)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Beste Tour", f"{a.gap:.1f} %", delta=f"letzte Tour {a.final_gap:.1f} %", delta_color="off", help="Abstand zur Schranke der kürzesten je besuchten Tour; im Delta der der letzten Tour der Kette.")
m2.metric("Ein Hill-Climbing-Abstieg", f"{a.hc_gap:.1f} %", delta=f"{_fmt_int(a.hc.evaluations)} Bewertungen", delta_color="off", help="Ein Abstieg (erste Verbesserung, dieselbe Nachbarschaft) aus derselben Startlösung.")
m3.metric("Hill Climbing mit Neustarts", f"{a.hcr_gap:.1f} %", delta=f"{a.hcr_starts} Abstiege, gleiches Budget", delta_color="off",
          help="So viele Abstiege aus zufälligen Startlösungen, wie ins Bewertungsbudget passen (der erste läuft immer zu Ende); die beste Tour zählt.")
m4.metric("Angenommene Verschlechterungen", f"{a.worse_fraction:.2%}", delta=f"{a.worse_share:.0%} der angenommenen Züge", delta_color="off", help="Anteil aller Vorschläge, die als Verschlechterung angenommen wurden; im Delta ihr Anteil an den angenommenen Zügen.")

if code == "beats_hc":
    st.success(f"✅ Besser als Hill Climbing bei gleichem Budget: {a.gap:.1f} % über der Schranke gegen {a.hcr_gap:.1f} % (Neustarts, {a.hcr_starts} Abstiege) und {a.hc_gap:.1f} % (ein Abstieg). Die Temperatur erlaubt es der Kette, Hügel zu überqueren, "
               "die der Abstieg nie verlässt. Andere Ketten streuen um dieses Ergebnis.")
elif code == "comparable":
    st.info(f"ℹ️ Gleichauf: Simulated Annealing {a.gap:.1f} %, Hill Climbing mit Neustarts {a.hcr_gap:.1f} % über der Schranke. Bei großem Budget und einer guten Nachbarschaft holen die Neustarts auf; eine andere Kette kann das Bild drehen.")
elif code == "hc_wins":
    st.warning(f"⚠️ Hill Climbing mit Neustarts ist besser: {a.hcr_gap:.1f} % gegen {a.gap:.1f} % über der Schranke bei gleichem Budget ({a.hcr_starts} Abstiege). Bei kleinen Instanzen oder zu kleinem Budget lohnt die Temperatur nicht – "
               "und eine einzelne Kette kann Pech haben.")
elif code == "too_cold":
    ctrl_word = {"metropolis": "die Temperatur", "threshold": "die Schwelle", "great_deluge": "der Wasserspiegel"}.get(settings.rule, "L")
    st.warning(f"⚠️ Zu kalt: nur {a.worse_fraction:.2%} der Vorschläge wurden als Verschlechterung angenommen – die Suche verhält sich wie ein Abstieg ({a.gap:.1f} % gegen {a.hc_gap:.1f} % für einen Hill-Climbing-Abstieg). "
               f"Erst wenn {ctrl_word} in den Bereich der typischen Verlängerungen kommt, entkommt sie den Hügeln.")
else:
    end_word = {"metropolis": "Die Endtemperatur", "threshold": "Die Endschwelle", "great_deluge": "Der Endabstand"}.get(settings.rule, "L")
    st.warning(f"⚠️ Das Ende ist zu heiß: die letzte Tour ({a.final_gap:.1f} %) ist {a.final_gap - a.gap:.1f} Prozentpunkte schlechter als die beste ({a.gap:.1f} %) – die Kette kommt nicht zur Ruhe, und die beste Tour ist nur ein Zufallstreffer im Rauschen. "
               f"{end_word} muss unter die typischen Verlängerungen der Nachbarn einer guten Tour sinken.")

d1, d2 = st.columns(2)
with d1:
    st.markdown("**Kennzahlen im Detail**")
    unit_time = lambda sec: f"{sec * 1000:.0f} ms"  # noqa: E731
    st.table({"": ["Länge (km)", "Abstand zur Schranke", "Bewertete Nachbarn", "Rechenzeit"],
              "Beste Tour": [f"{run.best_length:.1f}", f"{a.gap:.2f} %", _fmt_int(run.proposals), unit_time(a.seconds)],
              "Letzte Tour": [f"{run.final_length:.1f}", f"{a.final_gap:.2f} %", "–", "–"],
              "Beste + Abschlussabstieg": [f"{T.tour_length(a.polished_tour, a.D):.1f}", f"{a.polished_gap:.2f} %", "–", "–"],
              "Ein Abstieg": [f"{a.hc.length:.1f}", f"{a.hc_gap:.2f} %", _fmt_int(a.hc.evaluations), unit_time(a.hc_seconds)],
              "HC mit Neustarts": [f"{T.tour_length(a.hcr_tour, a.D):.1f}", f"{a.hcr_gap:.2f} %", _fmt_int(settings.budget), unit_time(a.hcr_seconds)]})
with d2:
    st.markdown("**Was gerechnet wurde**")
    if settings.rule == "lahc":
        ctrl_row, ctrl_val = "Listenlänge L", f"{settings.lahc_length:,}".replace(",", ".")
    elif settings.rule == "great_deluge":
        ctrl_row, ctrl_val = "Anfangs- → Endabstand (Einheiten)", f"{settings.gd_t0:.0f} → {settings.gd_t_end:.1f}"
    else:
        ctrl_row, ctrl_val = "T0 → T_end (Einheiten)", f"{temps_unit[0]:.2f} → {temps_unit[-1]:.3f}"
    st.table({"": ["Nachbarschaft", "Annahmeregel", "Abkühlplan", ctrl_row, "Budget", "Stufen", "Startlösung", "Kreuzungen der besten Tour"],
              "Einstellung": [C.NEIGHBORHOOD_LABELS[settings.neighborhood], C.RULE_LABELS[settings.rule], C.SCHEDULE_LABELS[settings.schedule] if settings.rule != "lahc" else "–", ctrl_val,
                              _fmt_int(settings.budget), f"{n_levels}", C.START_LABELS[settings.start], f"{a.crossings_end}"]})
    st.caption("Ein Vorschlag des Simulated Annealing ist ein bewerteter Nachbar, genau wie eine Bewertung im Hill Climbing – aber die Python-Schleife ist je Nachbar langsamer als die vektorisierte Bewertung aller Nachbarn im Abstieg (etwa um ein Mehrfaches (auf dem Entwicklungsrechner etwa das Dreifache)); "
               "die Rechenzeiten hängen vom Rechner ab, nur die Größenordnung zählt. Der Abschlussabstieg (2-opt auf der besten Tour) kostet wenig und hilft vor allem bei großen Instanzen: bei 200 Stopps und 200 Tausend Vorschlägen 7.7 → 7.0 %, bei 60 Stopps 1.4 → 1.4 %.")

st.markdown("---")

# --- Sweeps -----------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie stark hängt das Ergebnis von Budget, Temperatur, Plan und Nachbarschaft ab?")
sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", list(SWEEP_LABELS), format_func=lambda k: SWEEP_LABELS[k], key="sweep_select")
base_sweep = replace(settings, seed=0, chain_seed=0)
if st.button("Sweep über 5 feste Instanzen berechnen (dauert etwa 10 bis 60 Sekunden)", key="sweep_start"):
    st.session_state["sweep_done"] = st.session_state.get("sweep_done", set()) | {(sweep_param, base_sweep)}
if (sweep_param, base_sweep) in st.session_state.get("sweep_done", set()):
    with st.spinner("Rechne den Sweep über 5 feste Instanzen × 3 Ketten..."):
        rows_sweep = _sweep(sweep_param, base_sweep)
    categorical = sweep_param in ("schedule", "neighborhood", "start")
    labels = {"schedule": C.SCHEDULE_LABELS, "neighborhood": C.NEIGHBORHOOD_LABELS, "start": C.START_LABELS}.get(sweep_param)
    st.plotly_chart(build_sweep(rows_sweep, SWEEP_LABELS[sweep_param], categorical=categorical, key_labels=labels), width="stretch", key="sweep_chart")
    st.caption("Mittel und Streuung (Band bzw. Balken) über 5 feste Instanzen (Seeds 100000–100004, getrennt vom Seed oben) mit je drei Ketten; alle anderen Regler wie in der Seitenleiste. "
               "Die Streuung ist die Standardabweichung der Läufe, nicht des Mittels. Gestrichelt: ein Hill-Climbing-Abstieg (für diesen Sweep unabhängig vom durchgefahrenen Regler, außer bei Nachbarschaft, Stopps und Gruppen).")

st.markdown("---")

# --- Experimente ------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Budget: wann lohnt sich die Temperatur?")
if st.button("Budget von 10 Tausend bis 2 Millionen durchfahren (dauert etwa 30 Sekunden)", key="budget_start"):
    st.session_state["budget_on"] = True
if st.session_state.get("budget_on"):
    with st.spinner("Rechne 8 Budgets × 5 Instanzen × 3 Ketten..."):
        rows_b = _sweep("budget", base_sweep)
        dlb_rows_b = _dlb_budget(base_sweep) if settings.neighborhood == "2opt" else None
    st.plotly_chart(build_budget(rows_b, dlb_rows_b), width="stretch", key="budget_chart")
    table = {"Budget": [_fmt_int(r["value"]) for r in rows_b], "SA beste Tour (%)": [f"{r['gap']:.1f}" for r in rows_b], "SA letzte Tour (%)": [f"{r['final']:.1f}" for r in rows_b],
             "HC + Neustarts, voller Rescan (%)": [f"{r['hcr']:.1f}" for r in rows_b], "Abstiege": [f"{r['starts']:.1f}" for r in rows_b], "ein Abstieg (%)": [f"{r['hc']:.1f}" for r in rows_b]}
    if dlb_rows_b is not None:
        table["HC + Neustarts, Kandidatenliste + DLB (%)"] = [f"{r['gap']:.1f}" for r in dlb_rows_b]
        table["Abstiege (DLB)"] = [f"{r['starts']:.0f}" for r in dlb_rows_b]
    st.table(table)
    st.caption(f"Mittel über 5 feste Instanzen × 3 Ketten (Einstellungen wie in der Seitenleiste: {settings.n} Stopps, {C.NEIGHBORHOOD_LABELS[settings.neighborhood]}). Bei 60 Stopps, 2-opt und dem Standardplan: unter etwa 25 Tausend Vorschlägen ist Simulated Annealing nicht besser als ein Abstieg (10 Tausend: 8.0 % gegen 7.9 %); "
               "danach liegt es klar vorn gegenüber Hill Climbing mit vollem Rescan (25 Tausend: 4.1 %, 200 Tausend: 1.4 %, 1 Million: 0.7 %) – gegenüber Neustarts mit vollem Rescan (4.9 % bei 200 Tausend, 2.5 % bei 1 Million) schrumpft der Vorsprung mit dem Budget. "
               "Mit 2-opt + Or-opt schrumpft er noch weiter: bei 1 Million Vorschlägen 0.7 % gegen 1.0 % für Hill Climbing mit Neustarts (Hill Climbing hat hier 10 Abstiege). "
               "**Mit Kandidatenliste + Don't-Look-Bits (nur 2-opt gemessen) dreht sich das Bild:** dieselbe Güte braucht bei 60 Stopps nur noch rund 650 statt 74 000 bewertete Nachbarn, das Budget reicht dann für weit mehr Neustarts – Hill Climbing mit Neustarts liegt bei 200 Tausend bei **0.8 %** (Simulated Annealing 1.4 %, also **besser**) und bei 1 Million bei **0.7 %** (**gleichauf** mit Simulated Annealing). "
               "Der oben gezeigte Vorsprung von Simulated Annealing gilt also nur für die bewusst einfache, volle Rescan-Implementierung von Hill Climbing (siehe auch die [Hill-Climbing-Demo](https://sebastianhanisch-hill-climbing-demo.streamlit.app/)).")

st.markdown("---")

st.subheader("🔬 Annahmeregeln im Vergleich: braucht man den Zufall?")
st.caption("Alle vier Regeln mit ihren eigenen, kalibrierten Reglern (Threshold Accepting: T0/T_end wie in der Seitenleiste; Great Deluge: Anfangs-/Endabstand wie in der Seitenleiste; LAHC: Listenlänge L wie in der Seitenleiste) bei gleichem Budget.")
if st.button("Alle vier Regeln berechnen (dauert etwa 20 Sekunden)", key="rules_start"):
    st.session_state["rules_on"] = True
if st.session_state.get("rules_on"):
    with st.spinner("Rechne 4 Annahmeregeln × 5 Instanzen × 3 Ketten..."):
        rows_r = _rule_comparison(base_sweep, settings.budget)
    st.plotly_chart(build_rule_comparison(rows_r), width="stretch", key="rules_chart")
    st.table({"Regel": [r["label"] for r in rows_r], "Beste Tour (%)": [f"{r['gap']:.2f}" for r in rows_r], "Streuung (Prozentpunkte)": [f"{r['gap_sd']:.2f}" for r in rows_r],
              "HC + Neustarts (%)": [f"{r['hcr']:.2f}" for r in rows_r]})
    st.caption(f"Mittel über 5 feste Instanzen × 3 Ketten bei {_fmt_int(settings.budget)} Vorschlägen (60 Stopps, 2-opt, Standardregler): Metropolis 1.4 %, Late Acceptance Hill Climbing 2.2 %, Threshold Accepting 2.7 %, Great Deluge 3.2 % über der Schranke – "
               "Hill Climbing mit Neustarts bei gleichem Budget 4.9 %. Alle vier schlagen den Neustart-Abstieg klar; der Zufall in der Annahmeentscheidung (Metropolis) bringt gegenüber den drei deterministischen Regeln noch einen kleinen, aber sichtbaren Vorsprung. "
               "Die Reihenfolge zwischen den drei deterministischen Regeln ist keine Überraschung aus der Theorie, sondern eine Kalibrierungsfrage: LAHC braucht nur einen Regler (L) und ist am nächsten an Metropolis; Great Deluge hat den größten, am schwersten zu treffenden Regler (Anfangsabstand in absoluten Kilometern über der Schranke).")

st.markdown("---")

st.subheader("🔬 Das Temperaturfenster: Anfangs- gegen Endtemperatur")
if st.button("Anfangs- und Endtemperatur durchfahren (dauert etwa 20 Sekunden)", key="heatmap_start"):
    st.session_state["heatmap_on"] = True
if st.session_state.get("heatmap_on"):
    with st.spinner("Rechne 22 Temperaturpaare × 5 Instanzen × 2 Ketten..."):
        hm = _heatmap(base_sweep)
    st.plotly_chart(build_heatmap(hm), width="stretch", key="heatmap_chart")
    st.caption("Abstand der besten Tour zur Schranke (Mittel über 5 feste Instanzen × 2 Ketten, 100 Tausend Vorschläge, andere Regler wie in der Seitenleiste); leer: Endtemperatur über der Anfangstemperatur. "
               "Bei 60 Stopps und 2-opt: die Endtemperatur ist kritisch – bei 0.5 Einheiten 9.8 bis 22.4 %, bei Endtemperaturen bis 0.2 (und T0 ab 0.25) höchstens 3.8 %; die Anfangstemperatur ist weit: von 0.25 bis 2 liegt die Tour bei 0.02 bis 0.2 Endtemperatur zwischen 1.8 und 3.8 %. "
               "Zu kalt (T0 = 0.1): 6.3 bis 7.5 %, so schlecht wie ein Abstieg.")

st.markdown("---")

st.subheader("🔬 Streuung: wie verlässlich ist eine Kette?")
if st.button("20 Ketten auf dieser Instanz berechnen (dauert etwa 5 Sekunden)", key="spread_start"):
    st.session_state["spread_on"] = True
if st.session_state.get("spread_on"):
    with st.spinner("Rechne 20 Ketten und 20 Abstiege..."):
        sp = _spread(replace(settings, chain_seed=0))
    st.plotly_chart(build_spread(sp["sa"], sp["hc"]), width="stretch", key="spread_chart")
    s1, s2, s3 = st.columns(3)
    s1.metric("Simulated Annealing: Mittel ± Streuung", f"{sp['sa'].mean():.1f} ± {sp['sa'].std():.1f} %", help="Mittel und Standardabweichung des Abstands der besten Tour über 20 Ketten.")
    s2.metric("Hill Climbing: Mittel ± Streuung", f"{sp['hc'].mean():.1f} ± {sp['hc'].std():.1f} %", help="Ein Abstieg je Kette aus derselben zufälligen Startlösung.")
    s3.metric("Höchstens 2 % über der Schranke", f"{(sp['sa'] <= 2).mean():.0%} gegen {(sp['hc'] <= 2).mean():.0%}", help="Anteil der Ketten (Simulated Annealing) und Abstiege (Hill Climbing) mit höchstens 2 % Abstand.")
    st.caption("Dieselbe Instanz, 20 verschiedene Ketten-Seeds (die Startlösung wechselt mit). Bei den Standardeinstellungen: Simulated Annealing streut weniger als ein Abstieg, aber nicht null – "
               "eine einzelne Kette kann Pech haben. Standardinstanz (Seed 35): 1.5 ± 1.0 % (0.1 bis 5.1 %) gegen 5.0 ± 2.4 % (0.8 bis 11.5 %); 85 % der Ketten liegen höchstens 2 % über der Schranke, von den Abstiegen 5 %.")

st.markdown("---")

st.subheader("🔬 Skalierung: wie viel Budget braucht ein größeres Problem?")
if st.button("Stopps von 20 bis 200 durchfahren (dauert etwa 40 Sekunden)", key="scaling_start"):
    st.session_state["scaling_on"] = True
if st.session_state.get("scaling_on"):
    with st.spinner("Rechne 6 Größen × 2 Budgetregeln × 5 Instanzen × 3 Ketten..."):
        sc = _scaling(replace(base_sweep, n=C.DEFAULT_N))
    st.plotly_chart(build_scaling(sc), width="stretch", key="scaling_chart")
    st.caption("Mittel über 5 feste Instanzen × 3 Ketten (Einstellungen wie in der Seitenleiste außer Stopps und Budget). Bei gleichverteilten Stopps, 2-opt und dem Standardplan endet die beste Tour bei 20 / 40 / 60 / 100 / 150 / 200 Stopps mit **200 Tausend Vorschlägen** bei 0.05 / 0.9 / 1.4 / 3.8 / 6.3 / 7.7 % "
               "über der Schranke, mit **5 000 Vorschlägen je Stopp** bei 0.05 / 0.9 / 1.4 / 2.0 / 3.8 / 4.1 %; ein Hill-Climbing-Abstieg bei 1.9 / 7.7 / 7.9 / 9.1 / 9.1 / 9.5 %. "
               "Ein Budget, das mit n wächst, hält die Lücke nicht konstant: das nötige Budget wächst stärker als linear. Bei 200 Stopps braucht der Abstieg 4 Millionen bewertete Nachbarn, Simulated Annealing mit einer Million kommt auf 4.1 %.")

st.markdown("---")

# --- Grenzen ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Temperatur passt zur Instanz** | Zu kalt (T0 = 0.05, T_end = 0.02 Einheiten) liegt die Tour **7.1 %** über der Schranke – nicht besser als ein Abstieg (7.9 %); Endtemperatur 0.5: die beste Tour 9.5 %, die letzte **27.2 %**. Nur die Endtemperatur ist eng (0.1 bis 0.2 gut), die Anfangstemperatur weit (0.25 bis 4). | **Iterated Local Search** (Störungsstärke statt Temperatur), **ALNS** (lernt, welcher Umbau sich lohnt) |
| **Das Budget reicht** | Bei 200 Tausend Vorschlägen liegt die Tour bei 60 Stopps **1.4 %**, bei 100 Stopps 3.8 %, bei 200 Stopps **7.7 %** über der Schranke (Abstieg 9.5 %); mit 5 000 Vorschlägen je Stopp 4.1 % bei 200 Stopps. Bei 10 Tausend Vorschlägen (60 Stopps) ist die Kette nicht besser als ein Abstieg (8.0 % gegen 7.9 %). | **Tabu Search**, **VNS**/**ALNS** (gezieltere Umbauten je Vorschlag) |
| **Die Nachbarschaft ist gut** | Die Temperatur ersetzt sie nicht: Tausch allein **22.6 %**, Or-opt allein 5.7 % gegen 2-opt **1.4 %** und 2-opt + Or-opt 1.2 %. | **VNS** (wechselt die Nachbarschaft systematisch), **ALNS** |
| **Die letzte Tour ist die beste** | Die Kette vergisst: die letzte Tour liegt im Mittel 0.4 Prozentpunkte hinter der besten (1.8 gegen 1.4 %), bei zu heißem Ende 17.7 (27.2 gegen 9.5 %). Gemerkt wird die beste, aber nicht genutzt. | **Tabu Search** (Gedächtnis), **ILS** (kehrt zur besten Tour zurück) |
| **Die Theorie trägt** | Der logarithmische Plan garantiert das Optimum in unendlicher Zeit – im endlichen Budget ist er **3.4 %** über der Schranke gegen 1.4 % beim geometrischen. | (kein Nachfolger in der Linie: das ist die Grenze jeder Garantie ohne Zeitgrenze) |
| **Ein Lauf genügt** | 20 Ketten auf einer Instanz liegen zwischen 0.1 und **5.1 %** über der Schranke (Standardabweichung 1.0 Prozentpunkte); 15 % der Ketten enden mehr als 2 % darüber. | **GRASP** und Wiederholungen (viele Läufe statt einer langen Kette), Hill Climbing mit Neustarts als Maßstab |
| **Der Ringpuffer kommt nach (LAHC)** | Bei $L$ groß relativ zum Budget hinkt die Historie der tatsächlichen Konvergenz hinterher: **L=500 2.2 %**, L=700 **15.4 %**, L=1000 **44.2 %**, L=3000 **147 %** – kein Fehler in der Formel, sondern eine echte Grenze der Regel bei diesem Budget. | Kein direkter Nachfolger; die Lehre gilt sinngemäß auch für andere Ringpuffer-/Gedächtnisverfahren (**Tabu Search**) |
| **Der Anfangsspiegel deckt die Startlücke (Great Deluge)** | Ein zu knapper Anfangsabstand ließe die Suche sofort feststecken; die Demo hebt ihn automatisch auf die Startlänge an – trotzdem bleibt das Ergebnis bei knapper Kalibrierung schwächer und wechselhafter (**3.2 %** kalibriert gegen streuender bei zu knappem Anfangsabstand). | (Kalibrierungsfrage, kein Verfahrensnachfolger) |
"""
)
st.caption(
    "Die Nachbarn der Trajektorien-Metaheuristiken-Linie (noch nicht gebaut): Iterated Local Search mit VNS und ALNS, Tabu Search und GRASP; mit dem genetischen Algorithmus der Populations-Linie ergäbe sich später ein Memetischer Algorithmus. "
    "Simulated Annealing ist die einfachste Erweiterung der Wurzel: eine Regel mehr, ein Regler mehr."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Problem.** Kürzeste Rundtour über $N = n+1$ Knoten mit euklidischen Entfernungen $d_{ij}$; $L(\pi)$ ist die Länge einer Tour $\pi$.

**Metropolis-Regel.** Ein Vorschlag $\pi \to \pi'$ mit $\Delta = L(\pi') - L(\pi)$ wird mit $\min\{1, e^{-\Delta/T}\}$ angenommen. Ist der Vorschlag symmetrisch ($q(\pi \to \pi') = q(\pi' \to \pi)$, hier: 2-opt, Or-opt und Tausch mit gleichverteilter Wahl der Kanten), ist bei festem $T$ die **Boltzmann-Verteilung**
$p_T(\pi) \propto e^{-L(\pi)/T}$ die stationäre Verteilung der Kette (Detailed Balance). Bei $T \to 0$ konzentriert sie sich auf die kürzesten Touren, bei $T \to \infty$ ist sie gleichverteilt – die Demo prüft beides in den Tests (Kette auf 6 Knoten, alle 60 Touren).

**Abkühlplan.** Stufe $k = 0, \dots, K-1$ mit gleichbleibendem Kontrollwert: geometrisch $T_k = T_0 (T_{\text{end}}/T_0)^{k/(K-1)}$, linear $T_k = T_0 + (T_{\text{end}} - T_0)\, k/(K-1)$, logarithmisch $T_k = T_0 \ln 2 / \ln(k+2)$. Nach Hajek konvergiert Simulated Annealing mit $T_k = c / \ln(k+2)$ in Wahrscheinlichkeit gegen ein globales Optimum,
wenn $c$ mindestens die größte Tiefe eines lokalen Minimums ist – in unendlicher Zeit.

**Threshold Accepting** (Dueck & Scheuer 1990). Derselbe Vorschlag $\pi \to \pi'$ wird angenommen, wenn $\Delta \le T_k$ – dieselbe Formel wie Metropolis' Temperaturvergleich, aber ohne Zufallszahl: keine Boltzmann-Verteilung, kein Konvergenzbeweis, nur eine deterministische Faustregel.

**Great Deluge** (Dueck 1993). Ein Vorschlag wird angenommen, wenn die **absolute** neue Tourlänge unter einem Wasserspiegel $B_k = w + T_k$ bleibt: $L(\pi') \le B_k$. Anders als bei den anderen drei Regeln zählt nicht die Änderung $\Delta$, sondern die Gesamtlänge – der Spiegel sinkt nach demselben Plan wie oben, aber $T_0$ muss die anfängliche Lücke $L(\pi_0) - w$ decken;
die Demo hebt den Anfangsspiegel automatisch auf $\max(w + t_0 \bar d,\, L(\pi_0))$ an, damit eine zu knapp gewählte Anfangslücke nicht zum sofortigen Stillstand führt.

**Late Acceptance Hill Climbing** (Burke & Bykov 2012/2017). Ringpuffer $h_0, \dots, h_{L-1}$ mit Tourlängen, initial $L(\pi_0)$. Beim $k$-ten Vorschlag, $v = k \bmod L$: angenommen, wenn $L(\pi') \le h_v$ **oder** $L(\pi') \le L(\pi)$; danach $h_v \leftarrow L(\pi_{\text{neu}})$. Kein Plan, kein Temperaturbegriff – nur $L$. Bei zu großem $L$ (relativ zum Budget) hinkt der Puffer der tatsächlichen Konvergenz hinterher
und vergleicht noch lange mit einer sehr alten, schlechten Länge: die Suche nimmt dann fast jeden Vorschlag an und konvergiert nicht (gemessen: $L \le 600$ ist bei 60 Stopps/200 Tausend Vorschlägen sicher, ab $L \approx 700$ bricht die Güte ein).

**Längenänderung eines Vorschlags.** Wie im Hill Climbing aus wenigen Kanten: 2-opt $\Delta = d_{ac} + d_{bd} - d_{ab} - d_{cd}$; Or-opt (Stück $s_0 \dots s_1$ von $p$ nach $q$ zwischen $u$ und $v$ versetzen) $\Delta = d_{u s_0} + d_{s_1 v} - d_{uv} - (d_{p s_0} + d_{s_1 q} - d_{pq})$ (oder umgekehrt eingefügt); Tausch analog.

**Temperatureinheit.** $\bar d = w / N$ mit der 1-Baum-Schranke $w$; $T = t \cdot \bar d$ mit dem Regler $t$.

**Kennzahl.** Abstand zur Schranke $= 100 \cdot (L - w)/w$. **Hill Climbing mit Neustarts** bei gleichem Budget: Abstiege aus zufälligen Startlösungen, bis die bewerteten Nachbarn das Budget erreichen (der erste läuft immer zu Ende).

**Grenzen.** (1) Der Temperaturbereich muss zur Instanz passen (die Einheit nimmt die Kantenlänge ab, nicht die Struktur). (2) Das nötige Budget wächst stärker als linear mit $n$. (3) Die Nachbarschaft bleibt entscheidend. (4) Die Kette hat kein Gedächtnis außer der besten Tour. (5) Garantien gelten nur für unendliche Zeit.

Implementiert in `sa_algorithm.py` (Kette, Abkühlpläne), `sa_accept_rules.py` (die vier Annahmeregeln als reine Funktionen), `sa_tour.py` (Nachbarschaften, Abstieg, Schranke – aus der Hill-Climbing-Demo), `sa_scenario.py` (Instanzen), `sa_evaluation.py` (Kennzahlen, Sweeps, Experimente, Urteil).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
