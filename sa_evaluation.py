"""Auswertung der Simulated-Annealing-Demo: ein Lauf gegen Hill Climbing (ein Abstieg und Neustarts bei gleichem Bewertungsbudget), Sweeps, Vergleichstabellen, Kettenstreuung.

Der Abstand zur Schranke ist der Abstand zu einer *unteren* Schranke der kürzesten Tour (1-Baum, Held-Karp); er überschätzt die wahre Lücke um die Schrankenlücke (im Mittel unter 1 %).
Ein Vorschlag des Simulated Annealing ist ein bewerteter Nachbar - genauso zählt das Hill Climbing seine Bewertungen (dort wird nach jedem Zug neu von vorn gesucht: keine Nachbarschaftslisten)."""

import time
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

import sa_algorithm as SA
import sa_constants as C
import sa_dlb as DLB
import sa_scenario as S
import sa_tour as T


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    cluster_share: int = C.DEFAULT_BALLUNG
    seed: int = C.DEFAULT_SEED
    neighborhood: str = C.DEFAULT_NEIGHBORHOOD
    schedule: str = C.DEFAULT_SCHEDULE
    t0: float = C.DEFAULT_T0
    t_end: float = C.DEFAULT_T_END
    budget: int = C.DEFAULT_BUDGET
    levels: int = C.DEFAULT_LEVELS
    start: str = C.DEFAULT_START
    chain_seed: int = C.DEFAULT_CHAIN_SEED
    rule: str = C.DEFAULT_RULE
    lahc_length: int = C.DEFAULT_LAHC_L
    gd_t0: float = C.DEFAULT_GD_T0
    gd_t_end: float = C.DEFAULT_GD_T_END


@lru_cache(maxsize=256)
def instance(n, cluster_share, seed):
    inst = S.generate(n, cluster_share, seed)
    return inst, T.dist_matrix(inst.xy)


@lru_cache(maxsize=256)
def reference_bound(n, cluster_share, seed):
    """1-Baum-Schranke; Ziel des Subgradientenverfahrens ist die Länge eines guten lokalen Optimums (Nächster Nachbar + 2-opt/Or-opt, steilster Abstieg)."""
    inst, D = instance(n, cluster_share, seed)
    ref = T.descend(D, T.nearest_neighbor_tour(D), "2opt+oropt", "best", keep_steps=False)
    return T.held_karp_bound(D, ref.length, C.BOUND_ITERATIONS)


def unit_of(n, cluster_share, seed):
    """Temperatureinheit (km): mittlere Kantenlänge einer guten Tour = untere Schranke / Knotenzahl."""
    return reference_bound(n, cluster_share, seed) / (n + 1)


def make_start(settings, D):
    if settings.start == "nearest":
        return T.nearest_neighbor_tour(D)
    return T.random_tour(len(D), np.random.default_rng(settings.chain_seed))


def hill_climbing_restarts(D, budget, seed, neighborhood="2opt", rule="first"):
    """Hill Climbing mit Neustarts bei gleichem Bewertungsbudget: der erste Abstieg wird immer zu Ende geführt, weitere laufen nur mit dem Rest des Budgets. Gibt (beste Tour, Zahl der Starts, verbrauchte Bewertungen) zurück."""
    rng = np.random.default_rng(seed)
    used, starts, best = 0, 0, None
    while used < budget or best is None:
        cap = None if best is None else budget - used
        r = T.descend(D, T.random_tour(len(D), rng), neighborhood, rule, keep_steps=False, max_evaluations=cap)
        used += r.evaluations
        starts += 1
        if best is None or r.length < best.length:
            best = r
    return best.tour, starts, used


@dataclass
class Analysis:
    settings: Settings
    inst: object
    D: np.ndarray
    bound: float
    unit: float
    start_tour: np.ndarray
    run: object
    seconds: float
    hc: object                      # ein Hill-Climbing-Abstieg aus derselben Startlösung
    hc_seconds: float
    hcr_tour: np.ndarray            # Hill Climbing mit Neustarts bei gleichem Budget
    hcr_starts: int
    hcr_seconds: float
    crossings_end: int
    polished_tour: np.ndarray       # beste Tour nach einem Abschlussabstieg (2-opt, beste Verbesserung)

    def gap_of(self, length):
        return 100.0 * (length - self.bound) / self.bound

    @property
    def gap(self):
        return self.gap_of(self.run.best_length)

    @property
    def final_gap(self):
        return self.gap_of(self.run.final_length)

    @property
    def hc_gap(self):
        return self.gap_of(self.hc.length)

    @property
    def hcr_gap(self):
        return self.gap_of(T.tour_length(self.hcr_tour, self.D))

    @property
    def start_gap(self):
        return self.gap_of(T.tour_length(self.start_tour, self.D))

    @property
    def polished_gap(self):
        return self.gap_of(T.tour_length(self.polished_tour, self.D))

    @property
    def worse_fraction(self):
        """Anteil aller Vorschläge, die als Verschlechterung angenommen wurden."""
        return self.run.accepted_worse / max(self.run.proposals, 1)

    @property
    def last_worse_rate(self):
        """Anteil der Vorschläge der letzten Temperaturstufe, die als Verschlechterung angenommen wurden."""
        return float(self.run.worse_rate[-1])

    @property
    def worse_share(self):
        """Anteil der angenommenen Züge, die die Tour verlängerten."""
        return self.run.accepted_worse / max(self.run.accepted, 1)


def analyse(settings, keep_snapshots=True, with_hc=True):
    inst, D = instance(settings.n, settings.cluster_share, settings.seed)
    bound = reference_bound(settings.n, settings.cluster_share, settings.seed)
    unit = bound / len(D)
    start = make_start(settings, D)
    rule_t0, rule_t_end = (settings.gd_t0, settings.gd_t_end) if settings.rule == "great_deluge" else (settings.t0, settings.t_end)
    t0 = time.perf_counter()
    run = SA.anneal(D, start, settings.neighborhood, settings.schedule, rule_t0, rule_t_end, settings.budget, settings.levels, settings.chain_seed, unit, keep_snapshots=keep_snapshots,
                     rule=settings.rule, bound=bound, lahc_length=settings.lahc_length)
    seconds = time.perf_counter() - t0
    hc = hc_tour = None
    hc_seconds = hcr_seconds = 0.0
    hcr_starts = 0
    if with_hc:
        t0 = time.perf_counter()
        hc = T.descend(D, start, settings.neighborhood, "first", keep_steps=False)
        hc_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        hc_tour, hcr_starts, _ = hill_climbing_restarts(D, settings.budget, settings.chain_seed, settings.neighborhood)
        hcr_seconds = time.perf_counter() - t0
    polished = T.descend(D, run.best_tour, "2opt", "best", keep_steps=False).tour
    return Analysis(settings, inst, D, bound, unit, start, run, seconds, hc, hc_seconds, hc_tour, hcr_starts, hcr_seconds, T.count_crossings(inst.xy, run.best_tour), polished)


# --- Urteil ------------------------------------------------------------------------------------------------------------------------------------

HOT_END_GAP = 3.0                 # so viel schlechter darf die letzte Tour gegenüber der besten sein, bevor das Ende als zu heiß gilt (Prozentpunkte) ...
HOT_END_WORSE = 0.005             # ... und sofern in der letzten Stufe noch so viele Vorschläge als Verschlechterung angenommen werden
COLD_WORSE = 0.001                # unter diesem Anteil der Vorschläge, die als Verschlechterung angenommen werden, verhält sich die Suche wie ein Abstieg (zu kalt)
WIN_MARGIN = 1.0                  # so viel besser als Hill Climbing mit Neustarts gilt als Sieg (Prozentpunkte)
LOSE_MARGIN = 0.5                 # so viel schlechter als Hill Climbing mit Neustarts gilt als Niederlage


def verdict(a):
    """Code: too_hot (die letzte Tour ist deutlich schlechter als die beste und die Kette nimmt am Ende noch Verschlechterungen an: das Ende ist zu heiß), too_cold (fast keine Verschlechterung wird angenommen: die Suche ist ein Abstieg), beats_hc (deutlich besser als
    Hill Climbing mit Neustarts bei gleichem Budget), hc_wins (Hill Climbing mit Neustarts ist besser), comparable. Der Vergleich gilt für diesen einen Lauf - die Ketten streuen."""
    if a.final_gap - a.gap >= HOT_END_GAP and a.last_worse_rate >= HOT_END_WORSE:
        return "too_hot"
    if a.worse_fraction < COLD_WORSE:
        return "too_cold"
    if a.gap <= a.hcr_gap - WIN_MARGIN:
        return "beats_hc"
    if a.hcr_gap <= a.gap - LOSE_MARGIN:
        return "hc_wins"
    return "comparable"


# --- Sweeps und Tabellen -----------------------------------------------------------------------------------------------------------------------


def _mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def run_config(base, seeds=C.SWEEP_SEEDS, chains=C.SWEEP_CHAINS, **changes):
    """Mittel über die festen Instanzen und je `chains` Ketten-Seeds für die Einstellungen `base` mit `changes` (Instanz- und Ketten-Seed von `base` werden überschrieben)."""
    s0 = replace(base, **changes)
    rows = []
    for seed in seeds:
        for ch in range(chains):
            a = analyse(replace(s0, seed=seed, chain_seed=ch), keep_snapshots=False)
            rows.append({"gap": a.gap, "final": a.final_gap, "polished": a.polished_gap, "hc": a.hc_gap, "hcr": a.hcr_gap, "starts": a.hcr_starts, "seconds": a.seconds, "hc_seconds": a.hc_seconds,
                         "hcr_seconds": a.hcr_seconds, "worse_share": a.worse_share, "worse_fraction": a.worse_fraction, "accept": a.run.accepted / max(a.run.proposals, 1), "crossings": a.crossings_end, "hc_evaluations": a.hc.evaluations})
    out = {k: _mean(rows, k) for k in rows[0]}
    out.update({"gap_sd": float(np.std([r["gap"] for r in rows])), "hc_sd": float(np.std([r["hc"] for r in rows])), "gap_min": float(np.min([r["gap"] for r in rows])),
                "gap_max": float(np.max([r["gap"] for r in rows])), "n_runs": len(rows)})
    return out


SWEEP_VALUES = {"budget": (10000, 25000, 50000, 100000, 200000, 500000, 1000000, 2000000), "t0": (0.1, 0.25, 0.5, 1.0, 2.0, 4.0), "t_end": (0.005, 0.02, 0.05, 0.1, 0.2, 0.5),
                "schedule": SA.SCHEDULES, "neighborhood": SA.NEIGHBORHOODS, "levels": (1, 3, 10, 30, 100, 300, 1000), "n": (10, 20, 40, 60, 100, 150, 200), "cluster_share": (0, 25, 50, 75, 100),
                "start": ("random", "nearest")}
SWEEP_LABELS = {"budget": "Budget (Vorschläge)", "t0": "Anfangstemperatur T0", "t_end": "Endtemperatur", "schedule": "Abkühlplan", "neighborhood": "Nachbarschaft", "levels": "Temperaturstufen",
                "n": "Stopps", "cluster_share": "Anteil in Gruppen (%)", "start": "Startlösung"}


def sweep(param, base=Settings(), values=None):
    values = SWEEP_VALUES[param] if values is None else values
    return [{"value": v, **run_config(base, **{param: v})} for v in values]


HEAT_T0 = (0.1, 0.25, 0.5, 1.0, 2.0)
HEAT_T_END = (0.02, 0.05, 0.1, 0.2, 0.5)
HEAT_BUDGET = 100000
HEAT_CHAINS = 2


def heatmap_table(base=Settings()):
    """Abstand zur Schranke der besten Tour über Anfangs- und Endtemperatur (nur T_end <= T0), Budget HEAT_BUDGET, 5 Instanzen x HEAT_CHAINS Ketten. Zellen mit T_end > T0 sind None."""
    grid = []
    for t0 in HEAT_T0:
        row = []
        for te in HEAT_T_END:
            row.append(None if te > t0 else run_config(base, chains=HEAT_CHAINS, t0=t0, t_end=te, budget=HEAT_BUDGET)["gap"])
        grid.append(row)
    return {"t0": HEAT_T0, "t_end": HEAT_T_END, "gap": grid, "budget": HEAT_BUDGET}


SCALING_POLICIES = (("Budget 200 Tausend", lambda n: 200000), ("Budget 5 000 · Stopps", lambda n: 5000 * n))


def scaling_table(base=Settings()):
    return [{"label": label, "rows": [{"value": n, **run_config(base, n=n, budget=fn(n))} for n in C.SCALING_N]} for label, fn in SCALING_POLICIES]


def chain_spread(settings, k=C.SPREAD_CHAINS):
    """k Ketten-Seeds auf derselben Instanz: SA (beste Tour) und ein Hill-Climbing-Abstieg aus derselben zufälligen Startlösung."""
    sa_gaps, hc_gaps = [], []
    for ch in range(k):
        a = analyse(replace(settings, chain_seed=ch, start="random"), keep_snapshots=False)
        sa_gaps.append(a.gap)
        hc_gaps.append(a.hc_gap)
    return {"sa": np.array(sa_gaps), "hc": np.array(hc_gaps)}


# --- Annahmeregel im Vergleich -------------------------------------------------------------------------------------------------------------------
# Messreihe 2026-09-22 (5 Instanzen x 3 Ketten, Standardfall, Budget 200 Tausend): Metropolis 1.4 %, Late Acceptance Hill Climbing 2.2 %,
# Threshold Accepting 2.7 %, Great Deluge 3.2 % - Hill Climbing mit Neustarts bei gleichem Budget 4.9 %: alle vier Regeln schlagen den
# Neustart-Abstieg klar, der Zufall in der Annahmeentscheidung bringt gegenüber den drei deterministischen Regeln aber noch einen Vorsprung.


def rule_comparison_table(base=Settings(), budget=None, seeds=C.SWEEP_SEEDS, chains=C.SWEEP_CHAINS):
    """Alle vier Annahmeregeln bei gleichem Budget (Standard: `base.budget`), gemittelt über die festen Instanzen x `chains` Ketten - je einmal mit den zur Regel passenden Reglern."""
    budget = base.budget if budget is None else budget
    rows = []
    for rule in C.RULES:
        changes = {"rule": rule, "budget": budget}
        if rule == "great_deluge":
            changes.update(gd_t0=base.gd_t0, gd_t_end=base.gd_t_end)
        elif rule == "lahc":
            changes.update(lahc_length=base.lahc_length)
        out = run_config(base, seeds=seeds, chains=chains, **changes)
        rows.append({"rule": rule, "label": C.RULE_LABELS[rule], **out})
    return rows


# --- Kandidatenlisten + Don't-Look-Bits (sa_dlb.py) --------------------------------------------------------------------------------------------
# Nur 2-opt (siehe sa_dlb.py); zeigt, dass der Vergleich mit Hill Climbing oben nur für den vollen Rescan gilt
# (Messreihe 2026-09-22, project_trajectory_metaheuristics_dag_scoping.md): bei gleichem Budget schlägt Hill
# Climbing mit Neustarts und Kandidatenliste + Don't-Look-Bits Simulated Annealing knapp.


def dlb_restarts(D, cand, budget, seed):
    """Wie hill_climbing_restarts, mit dem Kandidatenlisten- + Don't-Look-Bit-Abstieg (sa_dlb.dlb_descend) statt dem vollen Rescan.
    Anders als hill_climbing_restarts gibt diese Funktion die beste Länge direkt zurück, nicht die Tour (hier nicht gebraucht)."""
    rng = np.random.default_rng(seed)
    used, starts, best = 0, 0, None
    while used < budget or best is None:
        cap = None if best is None else budget - used
        r = DLB.dlb_descend(D, T.random_tour(len(D), rng), cand, seed=starts, max_evaluations=cap)
        used += r.evaluations
        starts += 1
        if best is None or r.length < best:
            best = r.length
    return best, starts, used


def dlb_budget_sweep(base=Settings(), values=None, seeds=C.SWEEP_SEEDS, chains=C.SWEEP_CHAINS):
    """Wie sweep('budget', ...), aber Hill Climbing mit Neustarts über Kandidatenliste + Don't-Look-Bits statt vollem
    Rescan (nur 2-opt). Gleiche Budgetpunkte wie SWEEP_VALUES['budget']; gibt [{'value', 'gap', 'starts'}, ...] zurück."""
    values = SWEEP_VALUES["budget"] if values is None else values
    rows = []
    for budget in values:
        gaps, starts_l = [], []
        for seed in seeds:
            inst, D = instance(base.n, base.cluster_share, seed)
            bound = reference_bound(base.n, base.cluster_share, seed)
            cand = DLB.build_candidate_lists(D)
            for ch in range(chains):
                best, starts, _ = dlb_restarts(D, cand, budget, ch * 1000 + seed)
                gaps.append(100 * (best - bound) / bound)
                starts_l.append(starts)
        rows.append({"value": budget, "gap": float(np.mean(gaps)), "starts": float(np.mean(starts_l))})
    return rows
