"""Simulated Annealing für eine Rundtour (TSP), Python/numpy von Grund auf.

Je Iteration wird EIN zufälliger Nachbar der Tour vorgeschlagen (2-opt, Or-opt, Tausch oder eine Mischung), seine Längenänderung Delta aus wenigen Kanten berechnet und der Zug mit der Wahrscheinlichkeit
min(1, exp(-Delta / T)) angenommen (Metropolis-Regel). Ein Vorschlag ist ein bewerteter Nachbar - das Budget ist damit mit dem Hill Climbing vergleichbar, das jeden Nachbarn einzeln bewertet.
Die Temperatur ändert sich nur zwischen Stufen (Ketten gleicher Temperatur); sie wird in Vielfachen einer Einheit (der mittleren Kantenlänge einer guten Tour) angegeben."""

import math
from dataclasses import dataclass, field

import numpy as np

import sa_accept_rules as AR

SCHEDULES = ("geometric", "linear", "log")
NEIGHBORHOODS = ("swap", "2opt", "oropt", "2opt+oropt")
RULES = ("metropolis", "threshold", "great_deluge", "lahc")
MAX_SEGMENT = 3
CHUNK = 8192


def temperatures(schedule, t0, t_end, levels):
    """Temperatur je Stufe (`levels` Werte). geometric: T0 -> T_end mit festem Faktor; linear: T0 -> T_end gleichmäßig; log: T0 * ln(2) / ln(k + 2) (Hajek), erreicht T_end nicht."""
    k = np.arange(levels, dtype=float)
    if levels == 1:
        return np.array([float(t0)])
    if schedule == "geometric":
        return float(t0) * (float(t_end) / float(t0)) ** (k / (levels - 1))
    if schedule == "linear":
        return float(t0) + (float(t_end) - float(t0)) * k / (levels - 1)
    if schedule == "log":
        return float(t0) * math.log(2.0) / np.log(k + 2.0)
    raise ValueError(schedule)


@dataclass
class Run:
    best_tour: np.ndarray
    best_length: float
    final_tour: np.ndarray
    final_length: float
    temps: np.ndarray                     # roher Plan-Wert je Stufe (km, T0->T_end): Temperatur (Metropolis), Schwelle (Threshold Accepting),
                                           # Abstand über der Schranke (Great Deluge - die Wasserspiegel selbst sind bound + temps); bei LAHC ungenutzt
    accept_rate: np.ndarray               # Anteil angenommener Vorschläge je Stufe
    worse_rate: np.ndarray                # Anteil angenommener VERSCHLECHTERUNGEN an allen Vorschlägen je Stufe
    level_length: np.ndarray              # Länge der aktuellen Tour am Ende jeder Stufe
    level_best: np.ndarray                # beste Länge bis zum Ende jeder Stufe
    snapshots: list = field(default_factory=list)   # aktuelle Tour am Ende jeder Stufe
    proposals: int = 0
    accepted: int = 0
    accepted_worse: int = 0
    trace_iter: np.ndarray = None         # Vorschläge bis zu den Punkten der Verlaufskurve
    trace_length: np.ndarray = None       # aktuelle Länge dort
    trace_best: np.ndarray = None         # beste Länge dort
    rule: str = "metropolis"
    debug: list = field(default_factory=list)   # nur bei debug_trace=True: (delta, kontrollwert, angenommen) je Vorschlag - fürs Testen


def _kinds(neighborhood):
    if neighborhood == "swap":
        return ("swap",)
    if neighborhood == "2opt":
        return ("2opt",)
    if neighborhood == "oropt":
        return ("oropt",)
    if neighborhood == "2opt+oropt":
        return ("2opt", "oropt")
    raise ValueError(neighborhood)


def anneal(D, start, neighborhood="2opt", schedule="geometric", t0=1.0, t_end=0.05, budget=100000, levels=100, seed=0, unit=1.0, keep_snapshots=True, trace_points=300,
           rule="metropolis", bound=0.0, lahc_length=1000, debug_trace=False):
    """Ein Lauf. `t0` und `t_end` sind Vielfache von `unit` (km); `budget` = Zahl der gültigen Vorschläge (bewerteten Nachbarn), `levels` = Zahl der Stufen.
    `rule` bestimmt die Annahmeentscheidung (sa_accept_rules.py): 'metropolis' (Standard) nutzt t0/t_end/schedule als Temperatur; 'threshold' (Threshold Accepting)
    dieselben als deterministische Schwelle pro Zug; 'great_deluge' nutzt t0/t_end/schedule als Abstand ÜBER `bound` (der Wasserspiegel ist also `bound + Plan-Wert`,
    ein Vergleich der ABSOLUTEN Kandidatenlänge, nicht der Änderung); 'lahc' (Late Acceptance Hill Climbing) ignoriert t0/t_end/schedule/bound vollständig und
    braucht nur `lahc_length` (Ringpuffer-Länge in Vorschlägen)."""
    if neighborhood not in NEIGHBORHOODS:
        raise ValueError(neighborhood)
    if rule not in RULES:
        raise ValueError(rule)
    levels = max(1, min(int(levels), int(budget)))
    n = len(D)
    Dl = D.tolist()
    t = [int(x) for x in start]
    kinds = _kinds(neighborhood)
    n_kinds = len(kinds)
    length = sum(Dl[t[k]][t[(k + 1) % n]] for k in range(n))
    eff_t0 = max(t0 * unit, length - bound) if rule == "great_deluge" else t0 * unit   # Wasserspiegel darf nie unter die Startlänge fallen (sonst nimmt Great Deluge ab Vorschlag 1 fast nichts mehr an)
    temps = temperatures(schedule, eff_t0, t_end * unit, levels)
    level_len = budget // levels
    rng = np.random.default_rng(seed)
    best_length, best_tour = length, list(t)
    accept_rate, worse_rate = np.zeros(levels), np.zeros(levels)
    level_length, level_best = np.zeros(levels), np.zeros(levels)
    snapshots = []
    accepted = accepted_worse = proposals = 0
    trace_every = max(1, budget // trace_points)
    tr_it, tr_len, tr_best = [], [], []
    seg_choices = [L for L in range(1, MAX_SEGMENT + 1) if n >= L + 3]
    n_seg = len(seg_choices)
    exp = math.exp
    is_metropolis, is_threshold, is_deluge, is_lahc = (rule == "metropolis", rule == "threshold", rule == "great_deluge", rule == "lahc")
    debug = []
    lahc_L = max(1, int(lahc_length))
    lahc_history = [length] * lahc_L if is_lahc else None                 # Ringpuffer mit Tourlängen, initial = Startlänge (Burke & Bykov)
    for lvl in range(levels):
        T = float(temps[lvl])
        inv_t = 1.0 / T if T > 0 else float("inf")
        level_value = bound + T if is_deluge else T                      # Wasserspiegel (Great Deluge) bzw. Schwelle/Temperatur (die anderen)
        steps = level_len if lvl < levels - 1 else budget - level_len * (levels - 1)
        acc = accw = done = 0
        while done < steps:
            m = min(CHUNK, (steps - done) + (steps - done) // 8 + 16)
            r_kind = rng.integers(0, n_kinds, size=m).tolist()
            r_a = rng.integers(0, n, size=m).tolist()
            r_b = rng.integers(0, n, size=m).tolist()
            r_l = rng.integers(0, n_seg, size=m).tolist()
            r_o = rng.integers(0, 2, size=m).tolist()
            r_u = rng.random(size=m).tolist()
            r_f = rng.random(size=m).tolist()
            for q in range(m):
                if done >= steps:
                    break
                kind = kinds[r_kind[q]]
                if kind == "oropt":
                    i = r_a[q]
                    L = seg_choices[r_l[q]]
                    off = int(r_f[q] * (n - L - 1))                         # Ziel-Kante gleichverteilt unter den N-L-1 Kanten, die nicht am Stück hängen
                    mpos = (i + L + off) % n
                    reverse = L > 1 and r_o[q] == 1
                    s0, s1 = t[i], t[(i + L - 1) % n]
                    p, nx = t[(i - 1) % n], t[(i + L) % n]
                    u, v = t[mpos], t[(mpos + 1) % n]
                    gain = Dl[p][s0] + Dl[s1][nx] - Dl[p][nx]
                    add = (Dl[u][s1] + Dl[s0][v] if reverse else Dl[u][s0] + Dl[s1][v]) - Dl[u][v]
                    delta = add - gain
                else:
                    i, j = r_a[q], r_b[q]
                    if i > j:
                        i, j = j, i
                    if j < i + 2 or (i == 0 and j == n - 1):
                        continue                                            # ungültiges Paar: neu ziehen (zählt nicht als Vorschlag)
                    if kind == "2opt":
                        a, b, c, d = t[i], t[i + 1], t[j], t[(j + 1) % n]
                        delta = Dl[a][c] + Dl[b][d] - Dl[a][b] - Dl[c][d]
                    else:
                        x, y = t[i], t[j]
                        pi, ni, pj, nj = t[(i - 1) % n], t[(i + 1) % n], t[(j - 1) % n], t[(j + 1) % n]
                        delta = Dl[pi][y] + Dl[y][ni] + Dl[pj][x] + Dl[x][nj] - Dl[pi][x] - Dl[x][ni] - Dl[pj][y] - Dl[y][nj]
                v = proposals % lahc_L if is_lahc else 0                  # Ringpufferstelle für DIESEN Vorschlag (0-basiert, vor dem Zählerstand)
                done += 1
                proposals += 1
                candidate_length = length + delta
                if is_metropolis:
                    accept = AR.metropolis(delta, T, r_u[q])
                elif is_threshold:
                    accept = AR.threshold(delta, T)
                elif is_deluge:
                    accept = AR.great_deluge(candidate_length, level_value)
                else:
                    lahc_old = lahc_history[v]
                    accept = AR.lahc(candidate_length, lahc_old, length)
                if accept:
                    if delta > 0.0:
                        accw += 1
                    acc += 1
                    length += delta
                    if kind == "2opt":
                        t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
                    elif kind == "swap":
                        t[i], t[j] = t[j], t[i]
                    else:
                        seg = [t[(i + w) % n] for w in range(L)]
                        rest = [t[(i + L + w) % n] for w in range(n - L)]
                        if reverse:
                            seg.reverse()
                        t = rest[:off + 1] + seg + rest[off + 1:]
                    if length < best_length - 1e-9:
                        best_length = length
                        best_tour = list(t)
                if debug_trace:
                    debug.append((delta, lahc_old if is_lahc else level_value, accept))
                if is_lahc:
                    lahc_history[v] = length                              # Länge NACH der Entscheidung (angenommen oder nicht), wie bei Burke & Bykov
                if proposals % trace_every == 0:
                    tr_it.append(proposals)
                    tr_len.append(length)
                    tr_best.append(best_length)
        accepted += acc
        accepted_worse += accw
        accept_rate[lvl] = acc / max(steps, 1)
        worse_rate[lvl] = accw / max(steps, 1)
        level_length[lvl] = length
        level_best[lvl] = best_length
        if keep_snapshots:
            snapshots.append(np.array(t, dtype=np.int64))
    final = np.array(t, dtype=np.int64)
    final_length = float(sum(Dl[final[k]][final[(k + 1) % n]] for k in range(n)))
    best = np.array(best_tour, dtype=np.int64)
    return Run(best, float(sum(Dl[best[k]][best[(k + 1) % n]] for k in range(n))), final, final_length, temps, accept_rate, worse_rate, level_length, level_best, snapshots, proposals, accepted, accepted_worse,
               np.array(tr_it), np.array(tr_len), np.array(tr_best), rule, debug)
