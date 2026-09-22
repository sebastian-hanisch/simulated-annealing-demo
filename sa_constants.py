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

# --- Annahmeregel (Metropolis / Threshold Accepting / Great Deluge / Late Acceptance Hill Climbing) -------------------------------------------
# Kalibriert 2026-09-22 am Standardfall (60 Stopps, 2-opt, Budget 200 Tausend, 5 Instanzen x 3 Ketten, zufällige Startlösung):
#   LAHC: Guete steigt bis L~500-600 (Luecke 2.2-5.2 %), dann Kollaps (L=700: 15.4 %, L=1000: 44.2 %, L=3000: 147 %) - der Ringpuffer
#         "kommt nicht mehr nach": bei grossem L vergleicht ein Vorschlag noch lange mit einem sehr alten, schlechten Wert, wird also
#         fast immer angenommen, und die Suche haengt statt zu konvergieren. Kein Fehler in der Formel, sondern eine echte Eigenschaft -
#         die Grenze L <= 650 gilt fuer DIESES Budget/DIESE Instanzgroesse, nicht allgemein.
#   Great Deluge: Anfangsabstand muss den Startabstand zur Schranke sicher uebersteigen (bei zufaelliger Startloesung, n=60: ~250-280
#         Einheiten) - darunter (t0=280) kippt die Suche gelegentlich katastrophal (Streuung > Mittelwert). sa_algorithm.anneal() hebt
#         den Anfangsspiegel zusaetzlich automatisch auf die Startlaenge an, falls t0 zu klein gewaehlt wird. Kalibriert fuer n=60 -
#         bei deutlich groesseren Instanzen (n=200) reicht das Budget nicht, um mit dem Spiegel Schritt zu halten (Luecke >800 %,
#         siehe README "Was nicht funktioniert hat"); wie bei Metropolis werden t0/t_end nicht automatisch mit n skaliert.
RULES = ("metropolis", "threshold", "great_deluge", "lahc")
RULE_LABELS = {"metropolis": "Metropolis", "threshold": "Threshold Accepting", "great_deluge": "Great Deluge", "lahc": "Late Acceptance Hill Climbing"}
DEFAULT_RULE = "metropolis"
LAHC_MIN, LAHC_MAX, LAHC_STEP, DEFAULT_LAHC_L = 10, 2000, 10, 500
GD_T0_MIN, GD_T0_MAX, GD_T0_STEP, DEFAULT_GD_T0 = 50.0, 600.0, 10.0, 300.0
GD_T_END_MIN, GD_T_END_MAX, GD_T_END_STEP, DEFAULT_GD_T_END = 0.5, 50.0, 0.5, 2.0


def _preset(neighborhood="2opt", schedule="geometric", t0=DEFAULT_T0, t_end=DEFAULT_T_END, budget=DEFAULT_BUDGET, n=DEFAULT_N, levels=DEFAULT_LEVELS, rule=DEFAULT_RULE, lahc_length=DEFAULT_LAHC_L,
             gd_t0=DEFAULT_GD_T0, gd_t_end=DEFAULT_GD_T_END):
    return {"n": n, "ballung": DEFAULT_BALLUNG, "seed": DEFAULT_SEED, "neighborhood": neighborhood, "schedule": schedule, "t0": t0, "t_end": t_end, "budget": budget, "levels": levels, "start": DEFAULT_START,
            "chain_seed": DEFAULT_CHAIN_SEED, "rule": rule, "lahc_length": lahc_length, "gd_t0": gd_t0, "gd_t_end": gd_t_end}


PRESETS = {
    "Standardfall (Voreinstellung)": _preset(),
    "Zu kalt (wie Hill Climbing)": _preset(t0=0.05, t_end=0.02),
    "Zu heiß am Ende": _preset(t0=1.0, t_end=0.5),
    "Kleines Budget (25 Tausend)": _preset(budget=25000),
    "Großes Budget (1 Million)": _preset(budget=1000000),
    "2-opt + Or-opt": _preset(neighborhood="2opt+oropt"),
    "Logarithmischer Plan": _preset(schedule="log"),
    "Große Instanz (200 Stopps, 1 Million)": _preset(n=200, budget=1000000),
    "Late Acceptance Hill Climbing": _preset(rule="lahc"),
    "Threshold Accepting": _preset(rule="threshold"),
    "Great Deluge": _preset(rule="great_deluge"),
    "Great Deluge (Anfangsabstand zu klein)": _preset(rule="great_deluge", gd_t0=280.0),
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
    "Late Acceptance Hill Climbing": "Listenlänge L = 500, kein Plan, kein Zufall in der Annahme: die beste Tour liegt im Mittel 2.2 % über der Schranke – knapp hinter Metropolis (1.4 %), klar vor Hill Climbing mit Neustarts (4.9 %).",
    "Threshold Accepting": "Dieselben T0/T_end wie Metropolis, aber deterministisch (Δ ≤ Schwelle statt Zufall): 2.7 % über der Schranke – der Zufall in der Annahme bringt gegenüber der reinen Schwelle noch etwas.",
    "Great Deluge": "Anfangsabstand 300, Endabstand 2 Einheiten über der Schranke: 3.2 % – am schwächsten der vier Regeln, aber immer noch klar vor Hill Climbing mit Neustarts (4.9 %).",
    "Great Deluge (Anfangsabstand zu klein)": "Anfangsabstand 280 Einheiten – knapp unter dem typischen Startabstand einer zufälligen Tour (250 bis 280 Einheiten): die Demo hebt den Wasserspiegel automatisch auf die Startlänge an, sonst würde die Suche hier gelegentlich katastrophal kippen; "
        "trotzdem bleibt das Ergebnis schwächer und wechselhafter als mit dem kalibrierten Anfangsabstand (300) – gelegentlich gewinnt sogar Hill Climbing mit Neustarts.",
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
    "Late Acceptance Hill Climbing": {"beats_hc", "comparable", "hc_wins"},
    "Threshold Accepting": {"beats_hc", "comparable", "hc_wins"},
    "Great Deluge": {"beats_hc", "comparable", "hc_wins"},
    "Great Deluge (Anfangsabstand zu klein)": {"beats_hc", "comparable", "hc_wins"},
}
