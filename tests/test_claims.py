"""Jede Zahl in den Hilfetexten, Presets, Tabellen und Grenzen der App ist hier über die fünf festen Sweep-Instanzen (je drei Ketten-Seeds) belegt.
Die Kette ist bis auf die Erzeugung der Zufallszahlen deterministisch; die Bänder sind weit genug für Unterschiede zwischen numpy-Versionen und Plattformen (die Streuung einer Kette beträgt etwa einen Prozentpunkt,
das Mittel über 15 Läufe wackelt um etwa 0.3). Positive UND negative Aussagen: wo Simulated Annealing nicht besser ist als ein Abstieg oder als Hill Climbing mit Neustarts, steht das hier ebenso als Test wie dort, wo es gewinnt.
Rechenzeiten sind nur als Größenordnung geprüft."""

from functools import lru_cache

import numpy as np
import pytest

import sa_constants as C
import sa_evaluation as ev


@lru_cache(maxsize=None)
def _cfg(items):
    return ev.run_config(ev.Settings(), **dict(items))


def cfg(**kw):
    return _cfg(tuple(sorted(kw.items())))


def near(value, expected, tol):
    assert abs(value - expected) <= tol, f"{value:.3f} statt {expected}"


# --- Seitenleiste: Stopps, Gruppen ------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n,sa,hc", [(10, 0.0, 1.7), (20, 0.05, 1.9), (40, 0.9, 7.7), (60, 1.4, 7.9), (100, 3.8, 9.1), (150, 6.3, 9.1), (200, 7.7, 9.5)])
def test_stops_sweep_at_200_thousand_proposals(n, sa, hc):
    r = cfg(n=n)
    near(r["gap"], sa, 0.6 if n <= 60 else 1.0)
    near(r["hc"], hc, 0.8)
    assert r["gap"] <= r["hc"] + 0.3                                       # Simulated Annealing ist nie schlechter als ein Abstieg (bei n = 10: beide bei der Schranke)


@pytest.mark.parametrize("n,sa", [(20, 0.05), (40, 0.9), (60, 1.4), (100, 2.0), (150, 3.8), (200, 4.1)])
def test_stops_sweep_with_five_thousand_proposals_per_stop(n, sa):
    near(cfg(n=n, budget=5000 * n)["gap"], sa, 0.7 if n <= 60 else 1.0)


def test_the_gap_grows_with_the_size_even_with_a_budget_proportional_to_n():
    gaps = [cfg(n=n, budget=5000 * n)["gap"] for n in (40, 100, 200)]
    assert gaps[0] < gaps[1] < gaps[2] and gaps[2] > 2 * gaps[0]


@pytest.mark.parametrize("share,sa,hc", [(0, 1.4, 7.9), (25, 1.7, 8.0), (50, 1.5, 5.7), (75, 1.1, 4.1), (100, 1.9, 4.4)])
def test_cluster_share_sweep(share, sa, hc):
    r = cfg(cluster_share=share)
    near(r["gap"], sa, 0.7)
    near(r["hc"], hc, 0.9)
    assert r["gap"] < r["hc"] - 2.0


# --- Nachbarschaft, Plan, Temperatur -----------------------------------------------------------------------------------------------------


def test_neighborhood_help_numbers():
    swap, opt2, oropt, both = (cfg(neighborhood=k) for k in ("swap", "2opt", "oropt", "2opt+oropt"))
    near(swap["gap"], 22.6, 5.0)
    near(opt2["gap"], 1.4, 0.6)
    near(oropt["gap"], 5.7, 1.5)
    near(both["gap"], 1.2, 0.6)
    near(swap["hc"], 54.5, 4.0)
    near(opt2["hc"], 7.9, 0.8)
    near(oropt["hc"], 10.4, 1.5)
    near(both["hc"], 3.7, 0.9)
    assert abs(opt2["gap"] - both["gap"]) < 0.8 and opt2["hc"] - both["hc"] > 3.0        # 2-opt + Or-opt ändert bei Simulated Annealing kaum etwas, beim Hill Climbing halbiert es die Lücke
    assert swap["gap"] > 10 * opt2["gap"] and swap["gap"] < swap["hc"] / 1.8 and oropt["gap"] > 2 * opt2["gap"]


def test_schedule_help_numbers():
    geo, lin, log = cfg(), cfg(schedule="linear"), cfg(schedule="log")
    near(geo["gap"], 1.4, 0.6)
    near(lin["gap"], 1.3, 0.6)
    near(log["gap"], 3.4, 1.2)
    assert abs(geo["gap"] - lin["gap"]) < 0.8 and log["gap"] > geo["gap"] + 1.0


@pytest.mark.parametrize("t0,gap,accept", [(0.1, 3.8, 0.003), (0.25, 2.1, 0.006), (0.5, 1.4, 0.012), (1.0, 1.2, 0.030), (2.0, 1.7, 0.076), (4.0, 1.8, 0.152)])
def test_start_temperature_sweep(t0, gap, accept):
    r = cfg(t0=t0)
    near(r["gap"], gap, 0.7)
    near(r["accept"], accept, max(0.002, 0.15 * accept))


@pytest.mark.parametrize("te,best,final", [(0.005, 3.1, 3.1), (0.02, 2.3, 2.4), (0.05, 1.75, 2.0), (0.1, 1.4, 1.8), (0.2, 1.4, 3.7), (0.5, 9.5, 27.2)])
def test_end_temperature_sweep(te, best, final):
    r = cfg(t_end=te)
    near(r["gap"], best, 0.7 if te < 0.5 else 3.0)
    near(r["final"], final, 1.0 if te < 0.5 else 4.0)
    assert r["final"] >= r["gap"] - 1e-9


def test_the_end_temperature_is_critical_and_the_start_temperature_is_wide():
    assert cfg(t_end=0.5)["gap"] > 4 * cfg()["gap"]
    for t0 in (0.25, 0.5, 1.0, 2.0, 4.0):
        assert cfg(t0=t0)["gap"] < 3.0
    assert cfg(t0=0.1)["gap"] > cfg()["gap"] + 1.5


def test_acceptance_numbers_of_the_readme():
    std = cfg()
    near(std["accept"], 0.012, 0.004)
    near(std["worse_share"], 0.47, 0.08)
    near(std["worse_fraction"], 0.0056, 0.002)
    near(cfg(t0=0.05, t_end=0.02)["worse_fraction"], 0.0004, 0.0004)
    assert std["gap_min"] < 0.5                                             # der beste der 15 Läufe liegt nahe der Schranke (0.08 %)


def test_last_tour_lags_the_best_tour():
    r = cfg()
    near(r["final"], 1.8, 0.7)
    near(r["final"] - r["gap"], 0.4, 0.4)
    hot = cfg(t_end=0.5)
    near(hot["final"] - hot["gap"], 17.7, 4.0)


# --- Budget -----------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("budget,best,final,hcr,starts", [(10000, 8.0, 8.0, 7.9, 1.0), (25000, 4.1, 4.5, 7.9, 1.0), (50000, 2.6, 3.3, 7.9, 1.0), (100000, 2.6, 3.1, 7.9, 1.9),
                                                          (200000, 1.4, 1.8, 4.9, 3.2), (500000, 1.0, 1.8, 2.9, 7.3), (1000000, 0.7, 1.25, 2.5, 14.0), (2000000, 0.5, 1.2, 1.9, 27.5)])
def test_budget_sweep(budget, best, final, hcr, starts):
    r = cfg(budget=budget)
    near(r["gap"], best, 1.2 if budget <= 25000 else 0.7)
    near(r["final"], final, 1.2 if budget <= 25000 else 0.8)
    near(r["hcr"], hcr, 0.9)
    near(r["starts"], starts, max(0.4, 0.15 * starts))
    near(r["hc"], 7.9, 0.8)


def test_simulated_annealing_is_no_better_than_one_descent_at_ten_thousand_proposals_and_clearly_better_later():
    assert abs(cfg(budget=10000)["gap"] - cfg(budget=10000)["hc"]) < 1.5
    for budget in (25000, 50000, 100000, 200000, 500000, 1000000, 2000000):
        assert cfg(budget=budget)["gap"] < cfg(budget=budget)["hc"] - 2.0


def test_the_advantage_over_hill_climbing_with_restarts_shrinks_in_relative_terms():
    ratios = [cfg(budget=b)["hcr"] / max(cfg(budget=b)["gap"], 0.05) for b in (200000, 1000000, 2000000)]
    assert ratios[0] > 2.5 and ratios[1] > 2.5 and cfg(budget=2000000)["hcr"] > 2.5 * cfg(budget=2000000)["gap"]     # der absolute Vorsprung schrumpft (3.5 -> 1.8 -> 1.4 Prozentpunkte)
    assert cfg(budget=200000)["hcr"] - cfg(budget=200000)["gap"] > cfg(budget=2000000)["hcr"] - cfg(budget=2000000)["gap"]


def test_two_opt_plus_or_opt_at_a_large_budget():
    r = cfg(neighborhood="2opt+oropt", budget=1000000)
    near(r["gap"], 0.65, 0.5)
    near(r["hcr"], 0.96, 0.6)
    near(r["starts"], 10.0, 2.5)
    assert abs(r["gap"] - r["hcr"]) < 0.9                                  # gleichauf: hier holen die Neustarts auf


def test_levels_sweep():
    for levels, gap in ((1, 8.9), (3, 2.2), (10, 1.1), (30, 1.7), (100, 1.4), (300, 1.3), (1000, 1.6)):
        near(cfg(levels=levels)["gap"], gap, 2.0 if levels == 1 else 0.9)
    near(cfg(levels=1)["final"], 28.4, 5.0)
    assert cfg(levels=1)["gap"] > 3 * cfg(levels=100)["gap"] and max(cfg(levels=l)["gap"] for l in (10, 30, 100, 300, 1000)) < 2.5


def test_start_solution_sweep():
    rnd, nn = cfg(), cfg(start="nearest")
    near(rnd["gap"], 1.4, 0.6)
    near(nn["gap"], 2.0, 0.8)
    near(rnd["hc"], 7.9, 0.8)
    near(nn["hc"], 7.3, 1.0)
    assert nn["gap"] >= rnd["gap"] - 0.3                                   # die gute Startlösung bringt nichts


# --- Presets und Grenzen ------------------------------------------------------------------------------------------------------------------


def test_preset_help_numbers():
    std = cfg()
    near(std["gap"], 1.4, 0.6)
    near(std["final"], 1.8, 0.7)
    near(std["hc"], 7.9, 0.8)
    near(std["hcr"], 4.9, 0.9)
    cold = cfg(t0=0.05, t_end=0.02)
    near(cold["gap"], 7.1, 1.2)
    assert cold["gap"] > 0.85 * cold["hc"] - 1.0 and cold["worse_fraction"] < ev.COLD_WORSE
    hot = cfg(t0=1.0, t_end=0.5)
    near(hot["gap"], 15.1, 4.0)
    near(hot["final"], 26.7, 4.5)
    near(cfg(budget=25000)["gap"], 4.1, 1.2)
    big = cfg(budget=1000000)
    near(big["gap"], 0.7, 0.4)
    near(big["hcr"], 2.5, 0.8)
    near(big["starts"], 14, 3)
    both = cfg(neighborhood="2opt+oropt")
    near(both["gap"], 1.2, 0.6)
    near(both["hc"], 3.7, 0.9)
    near(both["hcr"], 3.0, 0.9)
    near(cfg(schedule="log")["gap"], 3.4, 1.2)


def test_large_instance_preset_and_its_cost_for_hill_climbing():
    r = cfg(n=200, budget=1000000)
    near(r["gap"], 4.1, 1.0)
    near(r["hc"], 9.5, 0.8)
    near(r["hc_evaluations"] / 1e6, 4.0, 0.4)
    assert r["gap"] < r["hc"] - 3.0


def test_final_descent_helps_at_large_instances_only():
    small, big = cfg(), cfg(n=200)
    near(big["gap"], 7.7, 1.0)
    near(big["polished"], 7.0, 1.0)
    assert big["polished"] < big["gap"] - 0.3 and abs(small["polished"] - small["gap"]) < 0.3


def test_proposal_cost_is_a_multiple_of_the_vectorised_evaluation_cost():
    r = cfg()
    per_proposal = r["seconds"] / C.DEFAULT_BUDGET
    per_evaluation = r["hc_seconds"] / r["hc_evaluations"]
    assert 1.5 <= per_proposal / per_evaluation <= 20.0                    # Größenordnung (auf dem Entwicklungsrechner etwa das Dreifache)


def test_temperature_unit_is_about_ten_kilometres_at_sixty_stops():
    for seed in C.SWEEP_SEEDS:
        assert 8.5 <= ev.unit_of(60, 0, seed) <= 12.0


def test_metropolis_arithmetic_of_the_intro():
    assert np.exp(-1.0) == pytest.approx(0.37, abs=0.005) and np.exp(-3.0) == pytest.approx(0.05, abs=0.001)


# --- Heatmap, Streuung ---------------------------------------------------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _heat():
    return ev.heatmap_table()


def test_heatmap_captions():
    h = _heat()
    cells = {(t0, te): v for t0, row in zip(h["t0"], h["gap"]) for te, v in zip(h["t_end"], row) if v is not None}
    hot = [v for (t0, te), v in cells.items() if te == 0.5]
    assert 8.0 <= min(hot) and max(hot) <= 26.0 and near(min(hot), 9.8, 2.5) is None and near(max(hot), 22.4, 4.0) is None
    good = [v for (t0, te), v in cells.items() if t0 >= 0.25 and te <= 0.2]
    assert 1.2 <= min(good) and max(good) <= 4.5
    near(min(good), 1.8, 0.8)
    near(max(good), 3.8, 0.9)
    cold = [v for (t0, te), v in cells.items() if t0 == 0.1]
    near(min(cold), 6.3, 1.5)
    near(max(cold), 7.5, 1.5)


@lru_cache(maxsize=None)
def _spread():
    return ev.chain_spread(ev.Settings())


def test_chain_spread_numbers():
    s = _spread()
    near(float(s["sa"].mean()), 1.5, 0.5)
    near(float(s["sa"].std()), 1.0, 0.45)
    near(float(s["sa"].min()), 0.1, 0.4)
    near(float(s["sa"].max()), 5.1, 1.6)
    near(float(s["hc"].mean()), 5.0, 1.2)
    near(float(s["hc"].std()), 2.4, 0.9)
    near(float(s["hc"].min()), 0.8, 0.8)
    near(float(s["hc"].max()), 11.5, 3.0)
    near(float((s["sa"] <= 2).mean()), 0.85, 0.12)
    near(float((s["hc"] <= 2).mean()), 0.05, 0.10)
    assert s["sa"].std() < s["hc"].std()


# --- Schranke gegen das echte Optimum (CP-SAT) --------------------------------------------------------------------------------------------


def test_bound_is_close_to_the_optimum_for_uniform_stops_and_simulated_annealing_can_reach_it():
    cp = pytest.importorskip("ortools.sat.python.cp_model")
    gaps = []
    best_sa = []
    for seed in C.SWEEP_SEEDS:
        inst, D = ev.instance(60, 0, seed)
        di = np.rint(D * 10000).astype(int)
        m = cp.CpModel()
        lits = {(i, j): m.NewBoolVar("") for i in range(61) for j in range(61) if i != j}
        m.AddCircuit([(i, j, l) for (i, j), l in lits.items()])
        m.Minimize(sum(int(di[i, j]) * l for (i, j), l in lits.items()))
        s = cp.CpSolver()
        s.parameters.max_time_in_seconds = 90
        s.parameters.num_workers = 4
        assert s.StatusName(s.Solve(m)) == "OPTIMAL"
        opt = s.ObjectiveValue() / 10000
        b = ev.reference_bound(60, 0, seed)
        assert b <= opt + 0.01
        gaps.append(100 * (opt - b) / opt)
        a = ev.analyse(ev.Settings(seed=seed, budget=1000000), keep_snapshots=False, with_hc=False)
        assert a.run.best_length >= opt - 0.01
        best_sa.append(100 * (a.run.best_length - opt) / opt)
    near(float(np.mean(gaps)), 0.5, 0.3)
    assert float(np.mean(best_sa)) < 0.8                                   # bei 1 Million Vorschlägen liegt die Kette im Mittel unter 0.8 % über dem echten Optimum
