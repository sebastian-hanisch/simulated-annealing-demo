"""Die vier Annahmeregeln als reine Funktionen (ein Vorschlag = ein Aufruf), einzeln testbar mit Handwerten.
Metropolis ist die einzige mit Zufall in der Entscheidung selbst; die anderen drei sind deterministisch -
das ist der eigentliche Kontrast dieses Stücks: braucht man den Zufall in der Annahmeregel überhaupt?

metropolis: T = Temperatur (Vielfache der mittleren Kantenlaenge), u = vorab gezogene Zufallszahl in [0,1).
threshold (Threshold Accepting, Dueck & Scheuer 1990): T = Schwelle fuer die Laengenaenderung EINES Zuges,
    dieselbe Skala wie Metropolis' Temperatur, nur ohne Zufall.
great_deluge (Dueck 1993): level = Wasserspiegel, eine ABSOLUTE Tourlaenge (nicht eine Aenderung) -
    ein Kandidat wird angenommen, wenn seine Laenge unter dem Spiegel liegt, unabhaengig von der aktuellen Tour.
lahc (Late Acceptance Hill Climbing, Burke & Bykov 2012/2017): history_value = die Tourlaenge, die vor L
    Vorschlaegen an dieser Ringpufferstelle stand; ein Kandidat wird angenommen, wenn er kuerzer ist als
    DIESER alte Wert ODER kuerzer als die aktuelle Tour - kein Plan, keine Temperatur, nur die Listenlaenge L."""

import math

EXP = math.exp


def metropolis(delta, temperature, u):
    return delta <= 0.0 or u < EXP(-delta / temperature)


def threshold(delta, temperature):
    return delta <= temperature


def great_deluge(candidate_length, level):
    return candidate_length <= level


def lahc(candidate_length, history_value, current_length):
    return candidate_length <= history_value or candidate_length <= current_length
