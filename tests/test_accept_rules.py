"""Die vier Annahmeregeln (sa_accept_rules.py) gegen Handwerte - reine Funktionen, kein Zufallslauf noetig."""

import math

import pytest

import sa_accept_rules as R


# --- Metropolis (Referenz: unveraendert aus sa_algorithm, hier nur die Formel selbst) -----------------------------------------------------------


def test_metropolis_accepts_every_improvement_and_zero_delta():
    assert R.metropolis(-5.0, 1.0, 0.999999) is True
    assert R.metropolis(0.0, 1.0, 0.999999) is True                        # delta == 0 ist keine Verschlechterung


def test_metropolis_probability_matches_the_formula():
    # delta = T: exp(-1) ~= 0.3679 - u knapp darunter nimmt an, knapp darueber lehnt ab
    p = math.exp(-1.0)
    assert R.metropolis(1.0, 1.0, p - 1e-6) is True
    assert R.metropolis(1.0, 1.0, p + 1e-6) is False
    # delta = 3T: exp(-3) ~= 0.0498
    p3 = math.exp(-3.0)
    assert R.metropolis(3.0, 1.0, p3 - 1e-6) is True
    assert R.metropolis(3.0, 1.0, p3 + 1e-6) is False


def test_metropolis_never_accepts_a_worse_move_when_u_is_one():
    assert R.metropolis(10.0, 0.5, 1.0 - 1e-9) is False


# --- Threshold Accepting: deterministisch, dieselbe Skala wie Metropolis' Temperatur -------------------------------------------------------------


@pytest.mark.parametrize("delta,T,expected", [(-1.0, 0.1, True), (0.0, 0.1, True), (0.1, 0.1, True), (0.10001, 0.1, False), (5.0, 0.1, False)])
def test_threshold_accepts_iff_delta_at_most_the_threshold(delta, T, expected):
    assert R.threshold(delta, T) is expected


def test_threshold_has_no_randomness():
    # dieselbe delta/T-Kombination muss immer dasselbe Ergebnis geben (keine u-Abhaengigkeit im Funktionskopf)
    assert R.threshold(0.05, 0.1) == R.threshold(0.05, 0.1) is True


# --- Great Deluge: absolute Laenge gegen einen Wasserspiegel, unabhaengig von der aktuellen Tour ---------------------------------------------


@pytest.mark.parametrize("length,level,expected", [(99.0, 100.0, True), (100.0, 100.0, True), (100.0001, 100.0, False), (500.0, 100.0, False)])
def test_great_deluge_accepts_iff_the_candidate_length_is_under_the_level(length, level, expected):
    assert R.great_deluge(length, level) is expected


def test_great_deluge_can_accept_an_uphill_move_below_the_level_and_reject_a_downhill_move_above_it():
    # ein Kandidat, der laenger als die aktuelle Tour ist, aber noch unter dem Spiegel liegt, wird angenommen
    current_implied_by_uphill_example = 90.0
    assert R.great_deluge(95.0, 100.0) is True                             # 95 > 90 (Verschlechterung), aber unter dem Spiegel
    # ein Kandidat, der kuerzer als die aktuelle Tour ist, aber ueber dem Spiegel liegt, wird abgelehnt
    assert R.great_deluge(105.0, 100.0) is False                           # 105 < eine noch laengere aktuelle Tour waere egal - ueber dem Spiegel zaehlt nicht


# --- LAHC: Vergleich mit einem alten Ringpufferwert ODER der aktuellen Laenge ------------------------------------------------------------------


def test_lahc_accepts_if_better_than_the_old_history_value_even_if_worse_than_current():
    assert R.lahc(candidate_length=50.0, history_value=60.0, current_length=45.0) is True   # 50 > 45 (schlechter als aktuell), aber < history


def test_lahc_accepts_if_better_than_current_even_if_worse_than_the_old_history_value():
    assert R.lahc(candidate_length=44.0, history_value=40.0, current_length=45.0) is True   # 44 > history (40), aber < aktuell (45)


def test_lahc_rejects_if_worse_than_both():
    assert R.lahc(candidate_length=61.0, history_value=60.0, current_length=45.0) is False


def test_lahc_accepts_ties():
    assert R.lahc(candidate_length=60.0, history_value=60.0, current_length=45.0) is True
    assert R.lahc(candidate_length=45.0, history_value=60.0, current_length=45.0) is True
