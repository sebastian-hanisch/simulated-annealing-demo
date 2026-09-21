"""Simulated Annealing: Längenänderung jedes Vorschlags (Länge im Lauf gegen neu gemessene Tourlänge), Abkühlpläne, Metropolis-Regel und Boltzmann-Verteilung (Stationärverteilung einer langen Kette bei
festem T auf einer Instanz mit 6 Knoten, alle 60 Touren), Grenzfälle T -> 0 und T -> unendlich, Determinismus; dazu der kopierte Hill-Climbing-Kern (Brute-Force-Aufzählung, Schranke gegen CP-SAT)."""

import itertools
import math

import numpy as np
import pytest

import sa_algorithm as SA
import sa_tour as T


def _instance(n_nodes, seed):
    rng = np.random.default_rng(seed)
    xy = rng.random((n_nodes, 2)) * 100
    return xy, T.dist_matrix(xy)


# --- Abkühlpläne ---------------------------------------------------------------------------------------------------------------------------------


def test_temperature_schedules():
    g = SA.temperatures("geometric", 8.0, 0.5, 5)
    assert g[0] == pytest.approx(8.0) and g[-1] == pytest.approx(0.5) and np.allclose(g[1:] / g[:-1], (0.5 / 8.0) ** 0.25)
    lin = SA.temperatures("linear", 8.0, 0.5, 4)
    assert np.allclose(lin, [8.0, 5.5, 3.0, 0.5])
    lg = SA.temperatures("log", 6.0, 0.0, 6)
    assert lg[0] == pytest.approx(6.0) and np.allclose(lg, 6.0 * math.log(2) / np.log(np.arange(6) + 2.0))
    assert (np.diff(g) < 0).all() and (np.diff(lin) < 0).all() and (np.diff(lg) < 0).all()
    assert SA.temperatures("geometric", 3.0, 0.1, 1).tolist() == [3.0]
    with pytest.raises(ValueError):
        SA.temperatures("cubic", 1.0, 0.1, 5)


def test_unknown_neighborhood_is_rejected():
    xy, D = _instance(10, 1)
    with pytest.raises(ValueError):
        SA.anneal(D, np.arange(10), "3opt", budget=100)


# --- Längenänderung, Budget, Buchführung -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("neighborhood", ["swap", "2opt", "oropt", "2opt+oropt"])
def test_length_kept_during_the_run_equals_the_recomputed_length_and_tours_stay_valid(neighborhood):
    xy, D = _instance(14, 3)
    r = SA.anneal(D, T.random_tour(14, np.random.default_rng(1)), neighborhood, "linear", 3.0, 3.0, budget=6000, levels=60, seed=2, unit=10.0)
    assert len(r.snapshots) == 60
    for lvl, snap in enumerate(r.snapshots):
        assert sorted(snap.tolist()) == list(range(14))
        assert r.level_length[lvl] == pytest.approx(T.tour_length(snap, D), abs=1e-7)
    assert r.final_length == pytest.approx(T.tour_length(r.final_tour, D)) and r.best_length == pytest.approx(T.tour_length(r.best_tour, D))
    assert r.level_length[-1] == pytest.approx(r.final_length, abs=1e-7)


def test_budget_counts_valid_proposals_exactly():
    xy, D = _instance(20, 2)
    for neighborhood in ("swap", "2opt", "oropt", "2opt+oropt"):
        r = SA.anneal(D, np.arange(20), neighborhood, budget=5003, levels=17, seed=1, unit=1.0)
        assert r.proposals == 5003 and 0 <= r.accepted_worse <= r.accepted <= 5003
        assert len(r.temps) == len(r.accept_rate) == len(r.snapshots) == 17


def test_best_is_never_longer_than_the_final_tour_or_the_start_and_is_monotone_over_levels():
    xy, D = _instance(25, 4)
    start = T.random_tour(25, np.random.default_rng(0))
    r = SA.anneal(D, start, "2opt", "geometric", 2.0, 0.1, 20000, 40, 3, 8.0)
    assert r.best_length <= r.final_length + 1e-9 and r.best_length <= T.tour_length(start, D)
    assert (np.diff(r.level_best) <= 1e-9).all() and (r.trace_best <= r.trace_length + 1e-9).all()
    assert r.level_best[-1] == pytest.approx(r.best_length)


def test_determinism_and_seed_dependence():
    xy, D = _instance(18, 5)
    s = T.random_tour(18, np.random.default_rng(0))
    a = SA.anneal(D, s, "2opt+oropt", budget=8000, levels=20, seed=4, unit=8.0)
    b = SA.anneal(D, s, "2opt+oropt", budget=8000, levels=20, seed=4, unit=8.0)
    c = SA.anneal(D, s, "2opt+oropt", budget=8000, levels=20, seed=5, unit=8.0)
    assert np.array_equal(a.best_tour, b.best_tour) and np.array_equal(a.accept_rate, b.accept_rate) and a.accepted == b.accepted
    assert not np.array_equal(a.level_length, c.level_length)


def test_levels_are_capped_by_the_budget():
    xy, D = _instance(12, 1)
    r = SA.anneal(D, np.arange(12), budget=5, levels=100, seed=0)
    assert len(r.temps) == 5 and r.proposals == 5


# --- Grenzfälle der Temperatur ------------------------------------------------------------------------------------------------------------


def test_zero_temperature_only_accepts_improvements_and_ends_in_a_local_optimum():
    xy, D = _instance(15, 6)
    start = T.random_tour(15, np.random.default_rng(2))
    r = SA.anneal(D, start, "2opt", "linear", 1e-9, 1e-9, budget=60000, levels=30, seed=1, unit=1.0)
    assert r.accepted_worse == 0 and (np.diff(r.level_length) <= 1e-9).all()
    assert T.is_local_optimum(D, r.final_tour, "2opt")
    assert r.best_length == pytest.approx(r.final_length)


def test_infinite_temperature_accepts_everything():
    xy, D = _instance(15, 7)
    r = SA.anneal(D, np.arange(15), "2opt+oropt", "linear", 1e9, 1e9, budget=4000, levels=4, seed=1, unit=1.0)
    assert r.accepted == r.proposals and r.accept_rate.min() == pytest.approx(1.0)


# --- Boltzmann-Verteilung -------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("neighborhood", ["swap", "2opt", "oropt", "2opt+oropt"])
def test_long_chain_at_fixed_temperature_visits_tours_in_proportion_to_boltzmann_weights(neighborhood):
    n = 6
    xy, D = _instance(n, 11)
    temp = 0.35 * float(np.mean(D[D > 0]))
    tours = {}
    for perm in itertools.permutations(range(1, n)):
        c = tuple(T.canonical(np.array([0, *perm])).tolist())
        tours[c] = T.tour_length(c, D)
    assert len(tours) == 60
    weights = {c: math.exp(-(L - min(tours.values())) / temp) for c, L in tours.items()}
    z = sum(weights.values())
    p = {c: w / z for c, w in weights.items()}
    levels = 40000
    r = SA.anneal(D, np.arange(n), neighborhood, "linear", temp, temp, budget=levels * 12, levels=levels, seed=3, unit=1.0)
    counts = {}
    for snap in r.snapshots[2000:]:
        c = tuple(T.canonical(snap).tolist())
        counts[c] = counts.get(c, 0) + 1
    total = sum(counts.values())
    tv = 0.5 * sum(abs(counts.get(c, 0) / total - p[c]) for c in p)
    assert tv < 0.06, tv                                                    # gemessen etwa 0.02; Stichprobenfehler bei 38 000 Zuständen etwa 0.02-0.03
    best = min(tours, key=tours.get)
    assert counts.get(best, 0) / total == pytest.approx(p[best], abs=0.03)


def test_metropolis_acceptance_rate_matches_the_formula_for_a_fixed_delta():
    # zwei Knoten-Abstände erzeugen genau einen Delta-Wert: Tausch auf 4 Punkten eines Rechtecks ist nicht möglich (n < 5), daher 2-opt auf einem Rechteck mit zwei Zusatzpunkten
    xy = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0], [2.0, 0.0], [2.0, 2.0]])
    D = T.dist_matrix(xy)
    tour = [0, 4, 1, 2, 5, 3]                                              # Rand des Rechtecks: kürzeste Tour
    assert T.is_local_optimum(D, tour, "2opt")
    deltas = T.neighbor_deltas(np.array(tour), D, "2opt")
    assert (deltas > 0).all()
    temp = 1.0
    r = SA.anneal(D, np.array(tour), "2opt", "linear", temp, temp, budget=1, levels=1, seed=0, unit=1.0)
    # bei einem Vorschlag wird höchstens angenommen; die erwartete Annahmequote über viele Ketten aus derselben Tour ist der Mittelwert der Metropolis-Wahrscheinlichkeiten
    expected = float(np.mean(np.exp(-deltas / temp)))
    hits = sum(SA.anneal(D, np.array(tour), "2opt", "linear", temp, temp, budget=1, levels=1, seed=s, unit=1.0).accepted for s in range(4000))
    assert hits / 4000 == pytest.approx(expected, abs=0.03)
    assert r.proposals == 1


# --- Kopierter Hill-Climbing-Kern ------------------------------------------------------------------------------------------------------------


def _brute_two_opt_min(t, D):
    t = list(t)
    n = len(t)
    base = T.tour_length(np.array(t), D)
    best = np.inf
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            u = t[:i + 1] + t[i + 1:j + 1][::-1] + t[j + 1:]
            best = min(best, T.tour_length(np.array(u), D) - base)
    return best


def test_copied_two_opt_matches_brute_force_and_descent_ends_in_a_local_optimum():
    xy, D = _instance(12, 8)
    t = T.random_tour(12, np.random.default_rng(3))
    assert T.neighbor_deltas(t, D, "2opt").min() == pytest.approx(_brute_two_opt_min(t, D), abs=1e-9)
    r = T.descend(D, t, "2opt", "first")
    assert T.is_local_optimum(D, r.tour, "2opt") and _brute_two_opt_min(r.tour, D) >= -1e-9
    lengths = [s.length for s in r.steps]
    assert all(b < a - 1e-12 for a, b in zip(lengths, lengths[1:]))


def test_descent_stops_at_the_evaluation_budget():
    xy, D = _instance(40, 2)
    start = T.random_tour(40, np.random.default_rng(1))
    full = T.descend(D, start, "2opt", "first", keep_steps=False)
    cut = T.descend(D, start, "2opt", "first", keep_steps=False, max_evaluations=full.evaluations // 3)
    assert cut.n_moves < full.n_moves and cut.evaluations >= full.evaluations // 3 and cut.length > full.length
    assert T.descend(D, start, "2opt", "first", keep_steps=False, max_evaluations=10 ** 9).n_moves == full.n_moves


def test_bound_against_brute_force_and_cp_sat():
    for seed in range(3):
        xy, D = _instance(8, seed)
        opt = min(T.tour_length([0, *p], D) for p in itertools.permutations(range(1, 8)))
        b = T.held_karp_bound(D, opt)
        assert b <= opt + 1e-9 and b >= 0.9 * opt
    cp = pytest.importorskip("ortools.sat.python.cp_model")
    xy, D = _instance(20, 1)
    di = np.rint(D * 10000).astype(int)
    m = cp.CpModel()
    lits = {(i, j): m.NewBoolVar("") for i in range(20) for j in range(20) if i != j}
    m.AddCircuit([(i, j, l) for (i, j), l in lits.items()])
    m.Minimize(sum(int(di[i, j]) * l for (i, j), l in lits.items()))
    s = cp.CpSolver()
    s.parameters.max_time_in_seconds = 60
    assert s.StatusName(s.Solve(m)) == "OPTIMAL"
    opt = s.ObjectiveValue() / 10000
    ref = T.descend(D, T.nearest_neighbor_tour(D), "2opt+oropt", "best", keep_steps=False)
    b = T.held_karp_bound(D, ref.length)
    assert b <= opt + 0.01 and (opt - b) / opt < 0.03 and ref.length >= opt - 0.01
