"""Orakel-Test für den Hill-Climbing-Kern (`sa_tour`): Nachbarschafts-Deltas, Auswahlregeln, Bewertungszähler, Kreuzungszählung und 1-Baum-Schranke
gegen einen ANDEREN Rechenweg - Tourlängen werden hier durch vollständiges Neuberechnen nach dem Zerschneiden/Umbauen der Tourliste bestimmt
(nicht über die Delta-Formeln der Matrizen), Kreuzungen exakt mit Brüchen, die Schranke gegen die exakte Held-Karp-DP (kein Solver nötig).
Kein Test vergleicht den Code mit sich selbst oder mit eingefrorenen Werten."""

import itertools
from fractions import Fraction

import numpy as np

import sa_tour as T

EPS = 1e-9


def _len(tour, Dl):
    n = len(tour)
    return sum(Dl[tour[k]][tour[(k + 1) % n]] for k in range(n))


def _neighbors(t, Dl, neighborhood):
    """(Art, neue Tour) aller Nachbarn in der dokumentierten Reihenfolge (swap/2opt: nach (i, j); or-opt: Länge, Stück i, Kante m der Tour)."""
    t = [int(x) for x in t]
    n = len(t)
    out = []
    if neighborhood == "swap":
        for i in range(n):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                u = t[:]
                u[i], u[j] = u[j], u[i]
                out.append(("swap", u))
    if neighborhood in ("2opt", "2opt+oropt"):
        for i in range(n):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                out.append(("2opt", t[:i + 1] + t[i + 1:j + 1][::-1] + t[j + 1:]))
    if neighborhood in ("oropt", "2opt+oropt"):
        for length in (1, 2, 3):
            if n < length + 3:
                continue
            for i in range(n):
                seg = [t[(i + k) % n] for k in range(length)]
                rest = [t[(i + length + k) % n] for k in range(n - length)]
                for q in sorted(range(len(rest) - 1), key=lambda q: (i + length + q) % n):
                    best = min(((_len(rest[:q + 1] + s + rest[q + 1:], Dl), rest[:q + 1] + s + rest[q + 1:]) for s in (seg, seg[::-1])), key=lambda x: x[0])
                    out.append((f"oropt{length}", best[1]))
    return out


def _instances(rng, trials, n_lo, n_hi):
    for _ in range(trials):
        n = int(rng.integers(n_lo, n_hi))
        # Zufallspunkte und Gitterpunkte (viele Gleichstände, doppelte Punkte)
        xy = rng.random((n, 2)) * 100 if rng.random() < 0.6 else rng.integers(0, 4, (n, 2)).astype(float)
        yield n, xy, T.dist_matrix(xy)


def test_neighborhood_deltas_selection_rules_and_evaluation_counts_match_a_full_recomputation_oracle():
    rng = np.random.default_rng(2024)
    for n, xy, D in _instances(rng, 45, 4, 10):
        Dl = D.tolist()
        t = rng.permutation(n)
        base = T.tour_length(t, D)
        for neighborhood in T.NEIGHBORHOODS:
            orc = _neighbors(t, Dl, neighborhood)
            od = sorted(_len(u, Dl) - base for _, u in orc)
            assert np.allclose(od, sorted(T.neighbor_deltas(t, D, neighborhood).tolist()), atol=1e-7), (neighborhood, n)
            move, count = T.find_move(t, D, neighborhood, "best")
            assert count == len(orc)
            if od and od[0] < -EPS:
                assert move is not None and abs(move[-1] - od[0]) < 1e-7
                new = T.apply_move(t, move)
                assert abs(_len(new.tolist(), Dl) - (base + move[-1])) < 1e-7
            else:
                assert move is None
            move, count = T.find_move(t, D, neighborhood, "first")
            first = next((k for k, (_, u) in enumerate(orc) if _len(u, Dl) - base < -EPS), None)
            if first is None:
                assert move is None and count == len(orc)
            else:
                assert move is not None and count == first + 1
                assert abs(move[-1] - (_len(orc[first][1], Dl) - base)) < 1e-7
                assert abs(_len(T.apply_move(t, move).tolist(), Dl) - (base + move[-1])) < 1e-7


def test_descent_ends_in_a_local_optimum_found_by_full_recomputation():
    rng = np.random.default_rng(7)
    for n, xy, D in _instances(rng, 12, 5, 10):
        Dl = D.tolist()
        for neighborhood in T.NEIGHBORHOODS:
            for rule in T.RULES:
                r = T.descend(D, rng.permutation(n), neighborhood, rule)
                length = _len(r.tour.tolist(), Dl)
                assert abs(r.length - length) < 1e-7
                assert all(_len(u, Dl) >= length - 1e-7 for _, u in _neighbors(r.tour, Dl, neighborhood))


def _proper_crossing(p, q, r, s):
    """Echtes Schneiden zweier Strecken: p + a (q-p) = r + b (s-r) mit 0 < a, b < 1, exakt in Brüchen."""
    px, py = map(Fraction, p)
    qx, qy = map(Fraction, q)
    rx, ry = map(Fraction, r)
    sx, sy = map(Fraction, s)
    d1x, d1y, d2x, d2y = qx - px, qy - py, sx - rx, sy - ry
    den = d1x * d2y - d1y * d2x
    if den == 0:
        return False
    a = ((rx - px) * d2y - (ry - py) * d2x) / den
    b = ((rx - px) * d1y - (ry - py) * d1x) / den
    return 0 < a < 1 and 0 < b < 1


def test_count_crossings_matches_an_exact_fraction_oracle():
    rng = np.random.default_rng(11)
    for _ in range(150):
        n = int(rng.integers(4, 12))
        xy = rng.integers(0, 6, (n, 2)).astype(float) if rng.random() < 0.6 else rng.random((n, 2))
        t = rng.permutation(n)
        expected = 0
        for i in range(n):
            for j in range(i + 1, n):
                a, b, c, d = t[i], t[(i + 1) % n], t[j], t[(j + 1) % n]
                if len({a, b, c, d}) == 4 and _proper_crossing(xy[a], xy[b], xy[c], xy[d]):
                    expected += 1
        assert T.count_crossings(xy, t) == expected
    bowtie = np.array([[0, 0], [1, 1], [1, 0], [0, 1]], float)
    assert T.count_crossings(bowtie, [0, 1, 2, 3]) == 1 and T.count_crossings(bowtie, [0, 2, 1, 3]) == 0


def _held_karp_optimum(Dl):
    """Exakte kürzeste Rundtour (Held-Karp-DP über Teilmengen), n <= 9."""
    n = len(Dl)
    if n == 1:
        return 0.0
    if n == 2:
        return 2 * Dl[0][1]
    m = n - 1
    dp = {(1 << j, j): Dl[0][j + 1] for j in range(m)}
    for mask in range(1, 1 << m):
        for j in range(m):
            cur = dp.get((mask, j))
            if cur is None:
                continue
            for k in range(m):
                if not (mask >> k) & 1:
                    key = (mask | 1 << k, k)
                    v = cur + Dl[j + 1][k + 1]
                    if v < dp.get(key, float("inf")):
                        dp[key] = v
    return min(dp[((1 << m) - 1, j)] + Dl[j + 1][0] for j in range(m))


def test_dynamic_programming_oracle_agrees_with_brute_force():
    rng = np.random.default_rng(3)
    for _ in range(10):
        n = int(rng.integers(3, 8))
        Dl = T.dist_matrix(rng.random((n, 2)) * 50).tolist()
        brute = min(_len([0, *p], Dl) for p in itertools.permutations(range(1, n)))
        assert abs(_held_karp_optimum(Dl) - brute) < 1e-9


def test_one_tree_bound_never_exceeds_the_exact_optimum_and_degenerate_sizes_are_handled():
    rng = np.random.default_rng(5)
    for n, xy, D in _instances(rng, 40, 1, 9):
        Dl = D.tolist()
        opt = _held_karp_optimum(Dl)
        upper = T.descend(D, np.arange(n), "2opt+oropt", "best", keep_steps=False).length if n > 3 else opt
        assert T.held_karp_bound(D, upper, 300) <= opt + 1e-7


# --- Simulated Annealing (sa_algorithm.anneal): vollständige Nachsimulation aus dem Zufallsstrom -------------------------------------------------------------
# Die Zufallszahlen werden wie in anneal() gezogen (Reihenfolge je Block: Art, a, b, Segmentlänge, Richtung, u, f), jeder Vorschlag aber auf einer TOURLISTE
# ausgeführt und seine Längenänderung durch Neurechnen der ganzen Tourlänge bestimmt. Annahmeregel, Ringpuffer (LAHC), Temperaturen je Stufe und Zähler
# kommen aus den Definitionen der Literatur/dem Docstring, nicht aus sa_accept_rules oder den Delta-Formeln des Codes.


import math

import pytest

import sa_algorithm as SA


def _temps_by_definition(schedule, t0, t_end, levels):
    if levels == 1:
        return [t0]
    out = []
    for k in range(levels):
        if schedule == "geometric":
            out.append(t0 * math.exp(math.log(t_end / t0) * k / (levels - 1)))
        elif schedule == "linear":
            out.append(t0 + (t_end - t0) * k / (levels - 1))
        else:
            out.append(t0 * math.log(2) / math.log(k + 2))
    return out


def _simulate(D, start, neighborhood, schedule, t0, t_end, budget, levels, seed, unit, rule, bound, lahc_length):
    """Rückgabe (Liste (delta, angenommen, Länge davor) je Vorschlag, Endtour, Endlänge, beste Länge, Temperaturen je Stufe, vollständig?).
    Trifft ein Kandidat den Vergleichswert bis auf Rundung (|Differenz| < 1e-9; z. B. LAHC kehrt zu einer Länge von vor L Schritten zurück), entscheidet
    Gleitkomma-Rauschen - dort bricht die Nachsimulation ab (vollständig? = False) und es wird nur das bis dahin Berechnete verglichen."""
    n = len(D)
    Dl = D.tolist()
    t = [int(x) for x in start]
    kinds = {"swap": ("swap",), "2opt": ("2opt",), "oropt": ("oropt",), "2opt+oropt": ("2opt", "oropt")}[neighborhood]
    length = _len(t, Dl)
    levels = max(1, min(levels, budget))
    eff_t0 = max(t0 * unit, length - bound) if rule == "great_deluge" else t0 * unit
    temps = _temps_by_definition(schedule, eff_t0, t_end * unit, levels)
    level_len = budget // levels
    rng = np.random.default_rng(seed)
    seg_choices = [L for L in range(1, 4) if n >= L + 3]
    history = [length] * lahc_length
    best = length
    log = []
    proposals = 0
    for lvl in range(levels):
        T_ = temps[lvl]
        steps = level_len if lvl < levels - 1 else budget - level_len * (levels - 1)
        done = 0
        while done < steps:
            m = min(8192, (steps - done) + (steps - done) // 8 + 16)
            r_kind = rng.integers(0, len(kinds), size=m).tolist()
            r_a = rng.integers(0, n, size=m).tolist()
            r_b = rng.integers(0, n, size=m).tolist()
            r_l = rng.integers(0, len(seg_choices), size=m).tolist()
            r_o = rng.integers(0, 2, size=m).tolist()
            r_u = rng.random(size=m).tolist()
            r_f = rng.random(size=m).tolist()
            for q in range(m):
                if done >= steps:
                    break
                kind = kinds[r_kind[q]]
                if kind == "oropt":
                    i, L = r_a[q], seg_choices[r_l[q]]
                    seg = [t[(i + w) % n] for w in range(L)]
                    rest = [t[(i + L + w) % n] for w in range(n - L)]
                    off = int(r_f[q] * (n - L - 1))
                    if L > 1 and r_o[q] == 1:
                        seg = seg[::-1]
                    new = rest[:off + 1] + seg + rest[off + 1:]
                else:
                    i, j = sorted((r_a[q], r_b[q]))
                    if j < i + 2 or (i == 0 and j == n - 1):
                        continue
                    new = t[:i + 1] + t[i + 1:j + 1][::-1] + t[j + 1:] if kind == "2opt" else [t[j] if k == i else t[i] if k == j else t[k] for k in range(n)]
                delta = _len(new, Dl) - length
                done += 1
                cand = length + delta
                if rule == "metropolis":
                    accept = delta <= 0 or r_u[q] < math.exp(-delta / T_)
                    tie = abs(delta) < 1e-9
                elif rule == "threshold":
                    accept = delta <= T_
                    tie = abs(delta - T_) < 1e-9
                elif rule == "great_deluge":
                    accept = cand <= bound + T_
                    tie = abs(cand - (bound + T_)) < 1e-9
                else:
                    slot = proposals % lahc_length
                    accept = cand <= history[slot] or cand <= length
                    tie = min(abs(cand - history[slot]), abs(delta)) < 1e-9
                if tie:
                    return log, t, _len(t, Dl), best, temps, False
                log.append((delta, accept, length))
                if accept:
                    t, length = new, cand
                    best = min(best, length)
                if rule == "lahc":
                    history[proposals % lahc_length] = length
                proposals += 1
    return log, t, _len(t, Dl), best, temps, True


@pytest.mark.parametrize("rule", ["metropolis", "threshold", "great_deluge", "lahc"])
def test_annealing_run_equals_a_full_resimulation_with_recomputed_lengths(rule):
    rng = np.random.default_rng(100 + len(rule))
    complete_runs = 0
    for _ in range(60):
        n = int(rng.integers(5, 13))
        xy = rng.random((n, 2)) * 100
        D = T.dist_matrix(xy)
        start = rng.permutation(n)
        neighborhood = str(rng.choice(["swap", "2opt", "oropt", "2opt+oropt"]))
        schedule = str(rng.choice(["geometric", "linear", "log"]))
        unit = float(rng.uniform(5, 15))
        budget, levels = int(rng.integers(30, 700)), int(rng.integers(1, 12))
        t0, t_end = (float(rng.uniform(100, 300)), float(rng.uniform(1, 30))) if rule == "great_deluge" else (float(rng.uniform(0.3, 2.0)), float(rng.uniform(0.02, 0.3)))
        bound = 0.5 * _len(start.tolist(), D.tolist()) if rule == "great_deluge" else 0.0
        L = int(rng.integers(1, 40))
        seed = int(rng.integers(0, 10**6))
        run = SA.anneal(D, start, neighborhood, schedule, t0, t_end, budget, levels, seed, unit, rule=rule, bound=bound, lahc_length=L, debug_trace=True)
        log, tour, length, best, temps, complete = _simulate(D, start, neighborhood, schedule, t0, t_end, budget, levels, seed, unit, rule, bound, L)
        assert run.proposals == budget and len(run.debug) == budget
        assert np.allclose(run.temps, temps, rtol=1e-9)
        k = len(log)
        assert [round(d, 7) for d, _, _ in run.debug[:k]] == [round(d, 7) for d, _, _ in log]          # jedes Delta, auch abgelehnter Vorschläge
        assert [a for _, _, a in run.debug[:k - 1]] == [a for _, a, _ in log[:k - 1]]                 # Annahme bis vor dem ersten Rundungs-Gleichstand
        if complete:
            assert [a for _, _, a in run.debug] == [a for _, a, _ in log]
            assert run.final_tour.tolist() == tour and abs(run.final_length - length) < 1e-7 and abs(run.best_length - best) < 1e-7
            assert run.accepted == sum(a for _, a, _ in log) and run.accepted_worse == sum(a and d > 0 for d, a, _ in log)
            lv = max(1, min(levels, budget))
            sizes = [budget // lv] * (lv - 1) + [budget - budget // lv * (lv - 1)]
            at = 0
            for i, size in enumerate(sizes):                      # angezeigte Kennzahlen: Annahme- und Verschlechterungsquote je Stufe
                chunk = log[at:at + size]
                at += size
                assert abs(run.accept_rate[i] - sum(a for _, a, _ in chunk) / size) < 1e-12
                assert abs(run.worse_rate[i] - sum(a and d > 0 for d, a, _ in chunk) / size) < 1e-12
            complete_runs += 1
        elif k > 1:
            assert k <= budget
    # die Nachsimulation läuft in den meisten Läufen vollständig durch (Gleichstände sind selten, bei LAHC häufiger)
    assert complete_runs >= 15
