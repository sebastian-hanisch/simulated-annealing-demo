"""Szenario (wortgleich aus der Hill-Climbing-Demo, eingefrorene Werte) und Auswertung (Kennzahlen, Urteil, Hill Climbing mit Neustarts, Sweeps, Streuung)."""

from dataclasses import replace

import numpy as np
import pytest

import sa_constants as C
import sa_evaluation as ev
import sa_scenario as S
import sa_tour as T


# --- Szenario ---------------------------------------------------------------------------------------------------------------------------------


def test_instance_shape_depot_and_area():
    inst = S.generate(60, 0, 3)
    assert inst.xy.shape == (61, 2) and inst.n == 60 and inst.n_nodes == 61
    assert inst.xy[0].tolist() == [50.0, 50.0]
    assert inst.xy.min() >= 0.0 and inst.xy.max() <= C.AREA


def test_instance_is_deterministic_seed_dependent_and_matches_the_frozen_hill_climbing_instance():
    a, b, c = S.generate(40, 25, 5), S.generate(40, 25, 5), S.generate(40, 25, 6)
    assert np.array_equal(a.xy, b.xy) and not np.array_equal(a.xy, c.xy)
    # eingefroren: dieselbe Instanz wie in der Hill-Climbing-Demo (Seed 100000, 60 Stopps, gleichverteilt) - Schrankenwert und Kenngrößen
    inst, D = ev.instance(60, 0, 100000)
    assert inst.xy[0].tolist() == [50.0, 50.0] and float(inst.xy[1:].sum()) == pytest.approx(float(S.generate(60, 0, 100000).xy[1:].sum()))
    assert ev.reference_bound(60, 0, 100000) == pytest.approx(618.76, abs=0.05)


def test_grouped_stops_lie_closer_together_than_uniform_ones():
    def mean_nn(share):
        vals = []
        for seed in range(10):
            xy = S.generate(80, share, seed).xy[1:]
            d = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
            np.fill_diagonal(d, np.inf)
            vals.append(d.min(axis=1).mean())
        return float(np.mean(vals))
    assert mean_nn(100) < 0.7 * mean_nn(0)


# --- Analyse ------------------------------------------------------------------------------------------------------------------------------------


def test_analysis_fields_are_consistent():
    a = ev.analyse(ev.Settings())
    run = a.run
    assert a.bound < run.best_length <= run.final_length + 1e-9 <= a.hc.length + 20 * 1e3
    assert a.gap == pytest.approx(100 * (run.best_length - a.bound) / a.bound) and a.final_gap >= a.gap - 1e-9
    assert a.polished_gap <= a.gap + 1e-9 and a.hc_gap > a.gap and a.start_gap > a.hc_gap
    assert a.unit == pytest.approx(a.bound / 61) and run.proposals == a.settings.budget
    assert 0 < a.worse_fraction < 0.05 and 0 < a.worse_share < 1


def test_analysis_is_deterministic_and_chain_seed_matters():
    s = ev.Settings(n=30, budget=20000)
    a, b, c = ev.analyse(s), ev.analyse(s), ev.analyse(replace(s, chain_seed=1))
    assert np.array_equal(a.run.best_tour, b.run.best_tour) and a.gap == b.gap
    assert a.gap != c.gap or not np.array_equal(a.run.final_tour, c.run.final_tour)


def test_nearest_neighbor_start_is_deterministic_and_random_start_follows_the_chain_seed():
    s = ev.Settings(n=25, budget=5000, start="nearest")
    assert np.array_equal(ev.analyse(s).start_tour, ev.analyse(replace(s, chain_seed=3)).start_tour)
    r = ev.Settings(n=25, budget=5000)
    assert not np.array_equal(ev.analyse(r).start_tour, ev.analyse(replace(r, chain_seed=1)).start_tour)


def test_without_hill_climbing_the_comparison_fields_are_empty_and_fast():
    a = ev.analyse(ev.Settings(n=20, budget=2000), with_hc=False)
    assert a.hc is None and a.hcr_tour is None and a.hcr_starts == 0


def test_hill_climbing_with_restarts_uses_at_least_one_full_descent_and_stays_near_the_budget():
    inst, D = ev.instance(40, 0, 100000)
    single = T.descend(D, T.random_tour(len(D), np.random.default_rng(0)), "2opt", "first", keep_steps=False)
    tour, starts, used = ev.hill_climbing_restarts(D, 1000, 0)                 # Budget unter einem Abstieg: ein Abstieg läuft zu Ende
    assert starts == 1 and used > 1000 and T.is_local_optimum(D, tour, "2opt")
    tour, starts, used = ev.hill_climbing_restarts(D, 5 * single.evaluations, 0)
    assert starts >= 3 and used <= 5 * single.evaluations + 2 * len(D) ** 2      # der letzte Abstieg wird auf den Rest des Budgets begrenzt
    best_single = ev.hill_climbing_restarts(D, 1, 0)[0]
    assert T.tour_length(tour, D) <= T.tour_length(best_single, D) + 1e-9


# --- Urteil -------------------------------------------------------------------------------------------------------------------------------------


def _fake(gap, final_gap, hc_gap, hcr_gap, worse_fraction, last_worse_rate=0.0):
    class F:
        pass
    f = F()
    f.gap, f.final_gap, f.hc_gap, f.hcr_gap, f.worse_fraction, f.last_worse_rate = gap, final_gap, hc_gap, hcr_gap, worse_fraction, last_worse_rate
    return f


def test_verdict_codes():
    hot = ev.HOT_END_WORSE * 2
    assert ev.verdict(_fake(2.0, 2.0 + ev.HOT_END_GAP, 8.0, 5.0, 0.01, hot)) == "too_hot"
    assert ev.verdict(_fake(2.0, 2.0 + ev.HOT_END_GAP, 8.0, 5.0, 0.01, 0.0)) == "beats_hc"          # große Lücke allein (Ausreißer der Kette) ist nicht zu heiß
    assert ev.verdict(_fake(2.0, 2.5, 8.0, 5.0, 0.01, hot)) == "beats_hc"                            # heißes Ende ohne Lücke zur besten Tour auch nicht
    assert ev.verdict(_fake(2.0, 2.5, 8.0, 5.0, ev.COLD_WORSE / 2)) == "too_cold"
    assert ev.verdict(_fake(2.0, 2.5, 8.0, 5.0, 0.01)) == "beats_hc"
    assert ev.verdict(_fake(4.0, 4.5, 8.0, 3.0, 0.01)) == "hc_wins"
    assert ev.verdict(_fake(3.5, 4.0, 8.0, 4.0, 0.01)) == "comparable"
    assert ev.verdict(_fake(2.0, 2.0 + ev.HOT_END_GAP, 8.0, 5.0, ev.COLD_WORSE / 2, hot)) == "too_hot"     # zu heiß hat Vorrang


def test_verdict_of_real_runs():
    assert ev.verdict(ev.analyse(ev.Settings(t0=0.05, t_end=0.02))) == "too_cold"
    assert ev.verdict(ev.analyse(ev.Settings(t0=1.0, t_end=0.5))) == "too_hot"
    assert ev.verdict(ev.analyse(ev.Settings())) == "beats_hc"


# --- Sweeps und Tabellen ------------------------------------------------------------------------------------------------------------------


def test_run_config_counts_runs_and_aggregates():
    r = ev.run_config(ev.Settings(n=20, budget=5000))
    assert r["n_runs"] == len(C.SWEEP_SEEDS) * C.SWEEP_CHAINS
    assert r["gap_min"] <= r["gap"] <= r["gap_max"] and r["gap_sd"] >= 0 and r["hc_evaluations"] > 0 and r["seconds"] > 0


def test_run_config_ignores_the_seeds_of_the_base_settings():
    a = ev.run_config(ev.Settings(n=15, budget=3000, seed=1, chain_seed=5))
    b = ev.run_config(ev.Settings(n=15, budget=3000, seed=999, chain_seed=0))
    assert all(a[k] == b[k] for k in a if not k.endswith("seconds"))


def test_sweep_values_labels_and_ordering():
    assert set(ev.SWEEP_VALUES) == set(ev.SWEEP_LABELS)
    rows = ev.sweep("neighborhood", ev.Settings(n=15, budget=5000))
    assert [r["value"] for r in rows] == list(ev.SA.NEIGHBORHOODS) and rows[0]["gap"] > rows[1]["gap"]      # Tausch schlechter als 2-opt
    rows = ev.sweep("budget", ev.Settings(n=20), (2000, 20000))
    assert rows[1]["gap"] <= rows[0]["gap"] + 1.0


def test_heatmap_table_shape_and_empty_cells():
    h = ev.heatmap_table(ev.Settings(n=12))
    assert len(h["gap"]) == len(ev.HEAT_T0) and all(len(r) == len(ev.HEAT_T_END) for r in h["gap"])
    for t0, row in zip(h["t0"], h["gap"]):
        for te, v in zip(h["t_end"], row):
            assert (v is None) == (te > t0)


def test_scaling_table_structure(monkeypatch):
    monkeypatch.setattr(C, "SCALING_N", (10, 20))
    tab = ev.scaling_table(ev.Settings(budget=3000))
    assert len(tab) == 2 and all([r["value"] for r in blk["rows"]] == [10, 20] for blk in tab) and tab[0]["label"] != tab[1]["label"]


def test_chain_spread_returns_one_value_per_chain_and_is_deterministic():
    a = ev.chain_spread(ev.Settings(n=15, budget=3000), 5)
    b = ev.chain_spread(ev.Settings(n=15, budget=3000), 5)
    assert len(a["sa"]) == len(a["hc"]) == 5 and np.array_equal(a["sa"], b["sa"]) and np.array_equal(a["hc"], b["hc"])


# --- Kandidatenlisten + Don't-Look-Bits (sa_dlb.py) ----------------------------------------------------------------------------------------


def test_dlb_restarts_uses_at_least_one_descent_and_is_far_cheaper_than_a_full_rescan_restart():
    inst, D = ev.instance(60, 0, 100000)
    cand = ev.DLB.build_candidate_lists(D)
    best, starts, used = ev.dlb_restarts(D, cand, 1000, 0)
    assert starts >= 1 and used >= 1000
    hcr_tour, starts_hcr, used_hcr = ev.hill_climbing_restarts(D, 200000, 0)         # gibt eine Tour zurück, nicht die Länge
    best_dlb, starts_dlb, used_dlb = ev.dlb_restarts(D, cand, 200000, 0)             # gibt die Länge zurück
    assert starts_dlb > 20 * starts_hcr                          # gemessen: ~300 gegen ~3 Starts bei gleichem Budget
    assert best_dlb <= T.tour_length(hcr_tour, D)


def test_dlb_budget_sweep_structure_and_monotonicity():
    rows = ev.dlb_budget_sweep(values=(25000, 200000), seeds=(100000, 100001), chains=2)
    assert [r["value"] for r in rows] == [25000, 200000]
    assert rows[1]["starts"] > rows[0]["starts"]
    assert rows[1]["gap"] <= rows[0]["gap"] + 1.0                 # mehr Budget wird nicht schlechter (bis auf Rauschen)
