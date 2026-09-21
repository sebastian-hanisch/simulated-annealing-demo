"""Konstanten der Simulated-Annealing-Demo: Szenario (wortgleich zur Hill-Climbing-Demo), Regler, Beschriftungen (Presets folgen nach den Messungen)."""

AREA = 100.0                     # Kantenlänge des Gebiets in km
N_CLUSTERS = 5
CLUSTER_SIGMA = 6.0              # Streuung einer Gruppe in km
CLUSTER_MARGIN = 12.0            # Gruppenmittelpunkte liegen mindestens so weit vom Rand entfernt
SWEEP_SEEDS = tuple(range(100000, 100005))
SWEEP_CHAINS = 3                 # Ketten-Seeds je Instanz in Sweeps und Vergleichstabellen
BOUND_ITERATIONS = 300

N_MIN, N_MAX, DEFAULT_N, N_STEP = 10, 200, 60, 5
BALLUNG_MIN, BALLUNG_MAX, DEFAULT_BALLUNG, BALLUNG_STEP = 0, 100, 0, 25
SEED_MAX = 999999
DEFAULT_SEED = 35
DEFAULT_CHAIN_SEED = 0
DEFAULT_NEIGHBORHOOD = "2opt"
DEFAULT_SCHEDULE = "geometric"
DEFAULT_T0 = 0.5                 # Vielfache der mittleren Kantenlänge einer guten Tour
DEFAULT_T_END = 0.1
DEFAULT_BUDGET = 200000
DEFAULT_LEVELS = 100
DEFAULT_START = "random"
T0_MIN, T0_MAX, T0_STEP = 0.05, 4.0, 0.05
T_END_MIN, T_END_MAX, T_END_STEP = 0.005, 1.0, 0.005
BUDGETS = (10000, 25000, 50000, 100000, 200000, 500000, 1000000, 2000000)
LEVEL_OPTIONS = (1, 3, 10, 30, 100, 300, 1000)
SCALING_N = (20, 40, 60, 100, 150, 200)
SPREAD_CHAINS = 20
STEP_LABELS = ("1 Instanz", "2 Temperatur", "3 Abkühlen", "4 Tour bei sinkender Temperatur", "5 Ergebnis")
SCHEDULE_LABELS = {"geometric": "Geometrisch", "linear": "Linear", "log": "Logarithmisch (Hajek)"}
NEIGHBORHOOD_LABELS = {"swap": "Tausch", "2opt": "2-opt", "oropt": "Or-opt", "2opt+oropt": "2-opt + Or-opt"}
START_LABELS = {"random": "Zufällig", "nearest": "Nächster Nachbar"}


def _preset(neighborhood="2opt", schedule="geometric", t0=DEFAULT_T0, t_end=DEFAULT_T_END, budget=DEFAULT_BUDGET, n=DEFAULT_N, levels=DEFAULT_LEVELS):
    return {"n": n, "ballung": DEFAULT_BALLUNG, "seed": DEFAULT_SEED, "neighborhood": neighborhood, "schedule": schedule, "t0": t0, "t_end": t_end, "budget": budget, "levels": levels, "start": DEFAULT_START,
            "chain_seed": DEFAULT_CHAIN_SEED}


PRESETS = {
    "Standardfall (Voreinstellung)": _preset(),
    "Zu kalt (wie Hill Climbing)": _preset(t0=0.05, t_end=0.02),
    "Zu heiß am Ende": _preset(t0=1.0, t_end=0.5),
    "Kleines Budget (25 Tausend)": _preset(budget=25000),
    "Großes Budget (1 Million)": _preset(budget=1000000),
    "2-opt + Or-opt": _preset(neighborhood="2opt+oropt"),
    "Logarithmischer Plan": _preset(schedule="log"),
    "Große Instanz (200 Stopps, 1 Million)": _preset(n=200, budget=1000000),
}
# Mittel über die fünf festen Sweep-Instanzen (Seeds 100000-100004, je drei Ketten-Seeds), 60 gleichverteilte Stopps, Abstand zur Schranke; Hill Climbing bei gleichem Bewertungsbudget
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "60 Stopps, 2-opt, geometrisch von 0.5 auf 0.1 mittlere Kantenlängen, 200 Tausend Vorschläge: die beste Tour liegt im Mittel 1.4 % über der Schranke (die letzte 1.8 %); ein Hill-Climbing-Abstieg endet bei 7.9 %, Hill Climbing mit Neustarts bei gleichem Budget bei 4.9 %.",
    "Zu kalt (wie Hill Climbing)": "Anfangstemperatur 0.05, Endtemperatur 0.02 mittlere Kantenlängen: fast keine Verschlechterung wird angenommen, die Suche verhält sich wie ein Abstieg - 7.1 % über der Schranke, kaum besser als ein Hill-Climbing-Abstieg (7.9 %).",
    "Zu heiß am Ende": "Endtemperatur 0.5 mittlere Kantenlängen: die Kette kommt nicht zur Ruhe. Die beste Tour liegt 15.1 % über der Schranke, die letzte 26.7 %.",
    "Kleines Budget (25 Tausend)": "Nur 25 Tausend Vorschläge (ein Hill-Climbing-Abstieg braucht bei 60 Stopps etwa 74 Tausend bewertete Nachbarn): 4.1 % über der Schranke, der Abstieg endet bei 7.9 %.",
    "Großes Budget (1 Million)": "1 Million Vorschläge: 0.7 % über der Schranke; Hill Climbing mit Neustarts (14 Abstiege) erreicht bei gleichem Budget 2.5 %.",
    "2-opt + Or-opt": "Beide Nachbarschaften (je Vorschlag zufällig eine): 1.2 % über der Schranke; Hill Climbing endet mit dieser Nachbarschaft bei 3.7 %, mit Neustarts bei gleichem Budget bei 3.0 %.",
    "Logarithmischer Plan": "Abkühlplan nach Hajek (T0 · ln 2 / ln(k + 2)), der in der Theorie das Optimum garantiert: 3.4 % über der Schranke statt 1.4 % beim geometrischen Plan - er kühlt zu langsam ab.",
    "Große Instanz (200 Stopps, 1 Million)": "200 Stopps, 1 Million Vorschläge: die beste Tour liegt im Mittel 4.1 % über der Schranke, ein Hill-Climbing-Abstieg endet bei 9.5 % (er braucht etwa 4 Millionen bewertete Nachbarn).",
}
# Urteile, die bei diesem Preset über verschiedene Instanzen und Ketten-Seeds vorkommen (jedes Preset wird über mehrere Instanzen x 2 Ketten gemessen)
PRESET_EXPECTED_BANDS = {
    "Standardfall (Voreinstellung)": {"beats_hc", "comparable", "hc_wins"},
    "Zu kalt (wie Hill Climbing)": {"too_cold"},
    "Zu heiß am Ende": {"too_hot"},
    "Kleines Budget (25 Tausend)": {"beats_hc", "comparable", "hc_wins"},
    "Großes Budget (1 Million)": {"beats_hc", "comparable"},
    "2-opt + Or-opt": {"beats_hc", "comparable", "hc_wins"},
    "Logarithmischer Plan": {"beats_hc", "comparable", "hc_wins"},
    "Große Instanz (200 Stopps, 1 Million)": {"beats_hc", "comparable"},
}
