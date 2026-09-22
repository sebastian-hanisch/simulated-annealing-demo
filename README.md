# Simulated Annealing – eine Lieferrunde, die Hügel überwindet – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-simulated-annealing-demo.streamlit.app/)**

Zweites Stück der **Trajektorien-Metaheuristiken-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning":
anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Verfahren – **Simulated Annealing** – an einem wachsenden Beispiel: dieselbe Rundtour wie in der [hill-climbing-demo](../hill-climbing-demo) (ein Depot, n Kundenstopps in einem 100 × 100-km-Gebiet), dieselben Nachbarschaften (2-opt, Or-opt, Tausch), dieselbe untere Schranke.
Geändert wird **eine Regel**: ein Zug, der die Tour um Δ km **verlängert**, wird trotzdem mit der Wahrscheinlichkeit e^(−Δ/T) angenommen. Die Temperatur T sinkt nach einem Abkühlplan; am Ende zählt die beste besuchte Tour.

**Einordnung in die Reihe (die Kanten des Graphen):** Simulated Annealing ist **eine** Antwort auf die Schwäche der Wurzel (Hill Climbing bleibt im ersten lokalen Optimum stecken), die anderen sind die übrigen Nachfolger. Die Linie hat **keinen Konvergenzpunkt**.
```
hill-climbing-demo (Wurzel: nur bergab, bleibt im ersten Optimum stecken)        [gebaut]
  ├─ simulated-annealing-demo (nimmt Verschlechterungen an, Abkühlplan)          [dieses Stück]
  ├─ Iterated Local Search → VNS → ALNS  (stört ein Optimum; wechselt die Nachbarschaft; lernt Umbauten)   [nicht gebaut]
  ├─ Tabu Search              (Gedächtnis gegen Rückwege)                        [nicht gebaut]
  └─ GRASP                    (randomisierte Konstruktion, viele Starts)         [nicht gebaut]
```
(Mit dem genetischen Algorithmus der Populations-Linie ergäbe sich später ein Memetischer Algorithmus.)

Ergebnis in Kürze: **bei gleichem Bewertungsbudget schlägt Simulated Annealing das Hill Climbing klar** – 60 Stopps, 200 Tausend Vorschläge: die beste Tour liegt im Mittel **1.4 %** über der Schranke, ein Hill-Climbing-Abstieg endet bei 7.9 %, **Hill Climbing mit Neustarts** (3.2 Abstiege, gleiches Budget) bei 4.9 %. Aber: der Temperaturbereich muss stimmen – **zu kalt** ist es ein Abstieg (7.1 %), **zu heiß am Ende** kommt die Kette nicht zur Ruhe (beste Tour 9.5 %, letzte 27.2 %) –,
der Vorsprung vor den Neustarts **schrumpft mit dem Budget** (mit 2-opt + Or-opt bei 1 Million Vorschlägen fast gleichauf: 0.7 % gegen 1.0 %), und **das nötige Budget wächst stärker als linear mit der Größe** (bei 200 Stopps 7.7 %, mit 5 000 Vorschlägen je Stopp 4.1 %).

| Frage | Ergebnis (60 gleichverteilte Stopps, 2-opt, geometrisch von 0.5 auf 0.1 mittlere Kantenlängen, 200 Tausend Vorschläge, 100 Stufen, zufällige Startlösung; Mittel über 5 feste Instanzen, Seeds 100000–100004, mit je 3 Ketten-Seeds; Abstand = Prozent über der 1-Baum-Schranke, die bei gleichverteilten 60 Stopps im Mittel 0.5 % unter dem Optimum liegt; Hill Climbing bei demselben Bewertungsbudget) |
|---|---|
| Standardfall | ✅ Beste Tour **1.4 %** (letzte 1.8 %; mit Abschlussabstieg 1.4 %) gegen **7.9 %** für einen Hill-Climbing-Abstieg und **4.9 %** für Hill Climbing mit Neustarts (3.2 Abstiege). Angenommen werden 1.2 % der Vorschläge, knapp die Hälfte davon Verschlechterungen (0.56 % aller Vorschläge) |
| **Budget** | ✅ Beste Tour bei 10 / 25 / 50 / 100 / 200 Tausend / 0.5 / 1 / 2 Millionen Vorschlägen **8.0 / 4.1 / 2.6 / 2.6 / 1.4 / 1.0 / 0.7 / 0.5 %** (letzte Tour 8.0 / 4.5 / 3.3 / 3.1 / 1.8 / 1.8 / 1.25 / 1.2 %); Hill Climbing mit Neustarts 7.9 / 7.9 / 7.9 / 7.9 / 4.9 / 2.9 / 2.5 / 1.9 %. ❌ Bei **10 Tausend** ist Simulated Annealing nicht besser als ein Abstieg (8.0 gegen 7.9 %; ein Abstieg braucht 74 Tausend Bewertungen) |
| **Temperaturfenster** | ⚠️ **Endtemperatur** kritisch: 0.005 / 0.02 / 0.05 / 0.1 / 0.2 / 0.5 Einheiten → beste Tour 3.1 / 2.3 / 1.75 / 1.4 / 1.4 / **9.5 %**, letzte Tour 3.1 / 2.4 / 2.0 / 1.8 / 3.7 / **27.2 %**. **Anfangstemperatur** weit: 0.1 / 0.25 / 0.5 / 1 / 2 / 4 → 3.8 / 2.1 / 1.4 / 1.2 / 1.7 / 1.8 %; angenommene Vorschläge 0.3 / 0.6 / 1.2 / 3.0 / 7.6 / 15.2 %. (Einheit = mittlere Kantenlänge einer guten Tour, bei 60 Stopps etwa 10 km) |
| **Zu kalt / zu heiß** | ❌ T0 = 0.05, T_end = 0.02: **7.1 %** – wie ein Abstieg (7.9 %), nur 0.04 % der Vorschläge sind angenommene Verschlechterungen. T0 = 1, T_end = 0.5: beste Tour **15.1 %**, letzte **26.7 %** |
| **Abkühlplan** | ➖ Geometrisch 1.4 %, linear 1.3 %: gleichwertig. ❌ **Logarithmisch** (Hajek, in unendlicher Zeit optimal) **3.4 %**: kühlt im endlichen Budget zu langsam ab |
| **Nachbarschaft** | ➖ Beste Tour: Tausch **22.6 %**, 2-opt 1.4 %, Or-opt 5.7 %, 2-opt + Or-opt **1.2 %** (Hill Climbing: 54.5 / 7.9 / 10.4 / 3.7 %). Die Temperatur ersetzt die Nachbarschaft nicht (Tausch bleibt schlecht), aber 2-opt allein reicht: die Kombination ändert kaum etwas (1.2 statt 1.4 %), beim Hill Climbing halbiert sie die Lücke |
| **Stufen** | ➖ 1 / 3 / 10 / 30 / 100 / 300 / 1000 Stufen: 8.9 / 2.2 / 1.1 / 1.7 / 1.4 / 1.3 / 1.6 %. Eine Stufe (konstante Temperatur, letzte Tour 28.4 %) ist Unsinn; ab drei Stufen ändert sich nichts Sicheres mehr |
| **Startlösung** | ➖ Zufällig 1.4 %, Nächster Nachbar 2.0 %: die heiße Anfangstemperatur vergisst die Startlösung, die gute bringt nichts (Hill Climbing: 7.9 / 7.3 %) |
| **Größe** | ❌ Beste Tour bei 10 / 20 / 40 / 60 / 100 / 150 / 200 Stopps mit **200 Tausend Vorschlägen**: 0.0 / 0.05 / 0.9 / 1.4 / 3.8 / 6.3 / **7.7 %** (Abstieg 1.7 / 1.9 / 7.7 / 7.9 / 9.1 / 9.1 / 9.5 %); mit **5 000 Vorschlägen je Stopp**: 0.05 / 0.9 / 1.4 / 2.0 / 3.8 / 4.1 %. Das nötige Budget wächst stärker als linear; bei 200 Stopps und 1 Million Vorschlägen 4.1 % gegen 9.5 % (der Abstieg braucht dafür etwa 4 Millionen bewertete Nachbarn) |
| Gruppierte Stopps | ✅ 1.4 / 1.7 / 1.5 / 1.1 / 1.9 % bei 0 / 25 / 50 / 75 / 100 % der Stopps in fünf Gruppen (Abstieg 7.9 / 8.0 / 5.7 / 4.1 / 4.4 %) |
| **Streuung** | ⚠️ 20 Ketten auf der Standardinstanz: **1.5 ± 1.0 %** (0.1 bis 5.1 %), 85 % höchstens 2 % über der Schranke; 20 Hill-Climbing-Abstiege aus denselben Startlösungen 5.0 ± 2.4 % (0.8 bis 11.5 %), 5 % höchstens 2 % darüber. Simulated Annealing streut weniger, aber nicht null |
| **Abschlussabstieg** | ➖ Ein 2-opt-Abstieg auf der besten Tour kostet wenig und hilft bei großen Instanzen: 200 Stopps, 200 Tausend Vorschläge 7.7 → 7.0 %; 60 Stopps 1.4 → 1.4 % |
| **Optimum** | ✅ Bei 1 Million Vorschlägen liegt die Kette auf den fünf Instanzen im Mittel unter 0.8 % über dem echten Optimum (CP-SAT); die beste von 15 Ketten mit 200 Tausend Vorschlägen liegt nur 0.08 % über der Schranke |
| Rechenzeit | ➖ Ein Vorschlag kostet in der Python-Schleife ein Mehrfaches (auf dem Entwicklungsrechner etwa das Dreifache) einer Bewertung im vektorisierten Abstieg; 200 Tausend Vorschläge dauern dort etwa 0.1 s. Nur die Größenordnung zählt |

## Was die Demo zeigt

1. **Simulated Annealing in Aktion** (Schritt-Slider + Abspielen): **Instanz** → **Temperatur** (die Annahmewahrscheinlichkeit e^(−Δ/T) für drei Temperaturen, dazu die Verlängerungen der Nachbarn der besten Tour) → **Abkühlen** (Temperatur und Annahmequote je Stufe; Länge der aktuellen und der besten Tour über die Vorschläge, mit Schranke, einem Abstieg und den Neustarts) →
   **Tour bei sinkender Temperatur** (Stufen-Regler und ▶️ Abkühlen abspielen: die aktuelle Tour, blass die beste darunter) → **Ergebnis** (die beste Tour der Kette neben der besten aus Hill Climbing mit Neustarts).
2. **Was die Kette gefunden hat:** beste und letzte Tour, ein Abstieg, Hill Climbing mit Neustarts (gleiches Budget), angenommene Verschlechterungen; Urteil (`too_hot` → `too_cold` → `beats_hc` → `hc_wins` → `comparable`), Detailtabelle, Abschlussabstieg.
3. **📐 Sweeps** über Budget, Anfangs- und Endtemperatur, Plan, Nachbarschaft, Stufen, Stopps, Gruppen und Startlösung (feste Instanzen ab 100000, drei Ketten je Instanz).
4. **🔬 Experimente auf Abruf:** Budget von 10 Tausend bis 2 Millionen (Simulated Annealing gegen Abstieg und Neustarts); **Temperaturfenster** (Anfangs- gegen Endtemperatur als Heatmap); **Streuung** über 20 Ketten; **Skalierung** von 20 bis 200 Stopps mit zwei Budgetregeln.
5. **🚧 Grenzen:** Tabelle "Annahme – was passiert – wer setzt an" (die Temperatur passt zur Instanz, das Budget reicht, die Nachbarschaft ist gut, die letzte Tour ist die beste, die Theorie trägt, ein Lauf genügt).

Regler: Stopps (10–200), Anteil der Stopps in Gruppen, **Nachbarschaft** (Tausch / 2-opt / Or-opt / 2-opt + Or-opt), **Abkühlplan** (geometrisch / linear / logarithmisch), **Anfangstemperatur**, **Endtemperatur** (beim logarithmischen Plan ausgeblendet, der Wert bleibt erhalten; über der Anfangstemperatur wird sie auf diese begrenzt),
**Budget** (10 Tausend bis 2 Millionen Vorschläge), **Temperaturstufen**, **Startlösung**, Seed der Instanz (+ 🎲), Seed der Kette (+ 🎲).

## Messwerte der Presets (Instanz-Seed 35, Ketten-Seed 0; sie prüfen sich mit Urteil-Bändern selbst)

| Preset | Beste Tour | Letzte Tour | + Abschlussabstieg | Ein Abstieg | HC mit Neustarts (Abstiege) | Urteil |
|---|---|---|---|---|---|---|
| Standardfall (Voreinstellung) | 1.3 % | 2.0 % | 1.0 % | 6.8 % | 3.9 % (4) | beats_hc |
| Zu kalt (wie Hill Climbing) | 1.0 % | 1.1 % | 1.0 % | 6.8 % | 3.9 % (4) | too_cold |
| Zu heiß am Ende | 12.1 % | 31.0 % | 3.1 % | 6.8 % | 3.9 % (4) | too_hot |
| Kleines Budget (25 Tausend) | 0.5 % | 0.5 % | 0.4 % | 6.8 % | 6.8 % (1) | beats_hc |
| Großes Budget (1 Million) | 0.6 % | 1.4 % | 0.6 % | 6.8 % | 1.2 % (15) | comparable |
| 2-opt + Or-opt | 0.5 % | 2.6 % | 0.1 % | 3.5 % | 0.9 % (3) | comparable |
| Logarithmischer Plan | 2.8 % | 3.5 % | 2.8 % | 6.8 % | 3.9 % (4) | beats_hc |
| Große Instanz (200 Stopps, 1 Million) | 5.7 % | 5.8 % | 5.3 % | 10.7 % | 10.7 % (1) | beats_hc |

Die Presets zeigen einzelne Instanzen und Ketten – auf dieser Instanz ist die Kette bei "Zu kalt" zufällig gut (1.0 %), obwohl das Urteil `too_cold` lautet (nur 0.04 % der Vorschläge sind angenommene Verschlechterungen); die Mittelwerte über fünf Instanzen × drei Ketten stehen in der Tabelle oben und in den Hilfetexten (zu kalt im Mittel 7.1 %).

## Modell und Verfahren

- **Instanz, Nachbarschaften, Abstieg, Schranke** (`sa_scenario.py`, `sa_tour.py`): wortgleiche Kopie aus der [hill-climbing-demo](../hill-climbing-demo) (per Test gegen eingefrorene Werte); neu ist nur das Bewertungsbudget `max_evaluations` im Abstieg.
- **Kette** (`sa_algorithm.py`, Python/numpy von Grund auf): je Vorschlag ein zufälliger Nachbar (2-opt: gleichverteiltes Paar von Kanten, Or-opt: Segmentlänge 1–3, Position, Ziel und Richtung gleichverteilt, Tausch, oder eine Mischung), Δ in O(1) aus wenigen Kanten, Metropolis-Regel, Zufallszahlen vorab gezogen; die beste Tour wird gemerkt.
  **Ein Vorschlag = ein bewerteter Nachbar**, wie das Hill Climbing seine Bewertungen zählt. Temperatur in **Vielfachen der mittleren Kantenlänge einer guten Tour** (untere Schranke geteilt durch die Zahl der Knoten), in Stufen: geometrisch, linear, logarithmisch (Hajek). **Verlauf**, Annahmequoten je Stufe und Momentaufnahmen je Stufe für die Darstellung.
- **Hill Climbing mit Neustarts** (`sa_evaluation.py`): Abstiege aus zufälligen Startlösungen, bis die bewerteten Nachbarn das Budget erreichen; der erste läuft immer zu Ende, weitere mit dem Rest des Budgets (die Demo bewertet nach jedem Zug alle Nachbarn neu – ohne Nachbarschaftslisten wäre Hill Climbing bei gleichem Budget stärker).
- **Auswertung** (`sa_evaluation.py`): Kennzahlen, Urteil, Sweeps über feste Instanzen × Ketten, Temperaturfenster, Skalierung, Streuung.

## Nachtrag (2026-09-22): der Vergleich gilt nur für dieses Hill Climbing

Der Vergleich mit Hill Climbing auf dieser Seite bewertet nach jedem Zug **alle** Nachbarn neu (keine Nachbarschaftslisten, keine Don't-Look-Bits – bewusst, siehe Grenzen-Tabelle unten und die der [hill-climbing-demo](../hill-climbing-demo)). Eine Messreihe (kein eigenes Demo-Stück; vor dem geplanten Lin-Kernighan-Stück) mit einem **Kandidatenlisten- und Don't-Look-Bit-2-opt** (Nachbarschaft auf die 5 nächsten Knoten je Stopp beschränkt, Warteschlange nur über Knoten mit geänderten Kanten) zeigt:
ein Abstieg erreicht bei 60 Stopps dieselbe Güte wie im Standardfall oben (≈7 % über der Schranke) mit nur noch **rund 650 statt 74 000 bewerteten Nachbarn – dem Hundertfachen weniger**. Bei gleichem Budget wie in der Tabelle oben (200 Tausend / 1 Million) reicht das für **312 / 1 561 Neustarts statt 3.2 / 14**, und Hill Climbing mit Neustarts erreicht damit **0.7 % / 0.6 %** über der Schranke – **knapp besser als Simulated Annealing** in dieser Tabelle (1.4 % / 0.7 %).
Die Kandidatenliste kostet Exaktheit (bei 60 Stopps sind nur noch 51 % der Abstiege echte 2-opt-Optima, gegen 100 % auf kleinen Testinstanzen), aber messbar keine Güte. **Die Kernaussage dieser Seite gilt also nur für die hier bewusst einfach gehaltene Hill-Climbing-Implementierung**, nicht für Hill Climbing an sich: der eigentliche Unterschied zwischen den beiden Läufen war nicht "Temperatur schlägt reines Verbessern", sondern "billige gegen teure Bewertungen".
Die Demo selbst bleibt unverändert (das ist bewusst die einfache, gut lesbare Fassung); Kandidatenlisten und Don't-Look-Bits sind das Thema eines späteren Stücks der Nachbarschafts-Linie.

## Was nicht funktioniert hat / Grenzen

- **Vorab-Vermutungen (vor dem Bau gemessen):** (1) "Bei gleichem Budget schlägt Simulated Annealing das Hill Climbing" – **bestätigt**, aber mit Einschränkung: nicht bei 10 Tausend Vorschlägen (ein Abstieg braucht 74 Tausend), und **bei 2-opt + Or-opt und 1 Million Vorschlägen fast gleichauf** mit Neustarts (0.7 % gegen 1.0 %).
  (2) "Die Temperatur ist der Regler, es gibt ein Fenster" – **bestätigt, aber asymmetrisch**: die Endtemperatur ist eng (0.5 Einheiten: 9.5 %), die Anfangstemperatur weit (0.25 bis 4 Einheiten alle unter 3 %). (3) "Der Abkühlplan macht wenig Unterschied" – **halb**: geometrisch und linear gleich (1.4 / 1.3 %), der logarithmische (theoretisch optimal) deutlich schlechter (3.4 %).
  (4) "SA + 2-opt ≈ SA + 2-opt/Or-opt" – **bestätigt** (1.4 / 1.2 %), im Gegensatz zum Hill Climbing (7.9 / 3.7 %): die Temperatur nimmt der Nachbarschaft viel von ihrer Bedeutung, aber Tausch allein bleibt schlecht (22.6 %). (5) "Die Streuung über Ketten ist kleiner als beim Hill Climbing" – bestätigt (1.0 gegen 2.4 Prozentpunkte), aber nicht null (0.1 bis 5.1 %).
  (6) "Das nötige Budget wächst mit n stärker als linear" – **bestätigt**: 5 000 Vorschläge je Stopp reichen nicht (0.9 % bei 40, 4.1 % bei 200 Stopps). (7) "Die Kette braucht weniger Rechenzeit je Nachbar als der vektorisierte Abstieg" – **falsch**: je Vorschlag ein Mehrfaches (etwa das Dreifache).
  Nicht vorhergesagt: die **Anzahl der Stufen** ist fast gleichgültig (ab 3), die **Startlösung** ist es ebenfalls, und ein **Abschlussabstieg** auf der besten Tour hilft nur bei großen Instanzen (7.7 → 7.0 %).
- **Das Urteil vergleicht einen Lauf:** die Ketten streuen um etwa einen Prozentpunkt, ein einzelner Vergleich mit Hill Climbing kann kippen (die Presets haben deshalb Bänder). `too_cold` misst das Verhalten der Kette (Anteil der angenommenen Verschlechterungen), nicht den Vergleich mit einer zufälligen Ziehung.
- **Synthetische Instanzen:** euklidisch, gleichverteilt oder in fünf Gruppen, ein Fahrzeug, keine Kapazitäten oder Zeitfenster. Die Temperatur wird in Vielfachen der mittleren Kantenlänge angegeben; die Zahlen gelten für dieses Szenario. Zeiten hängen vom Rechner und der Python-Version ab (die Tests prüfen nur Größenordnungen).

## Verifikation

- **Kette:** die im Lauf mitgeführte Länge gegen die **neu gemessene Tourlänge jeder Momentaufnahme** (alle vier Nachbarschaften, auch die Tausch- und Or-opt-Sonderfälle); Budget = Zahl der gültigen Vorschläge; beste Tour nie länger als die letzte, monoton über die Stufen; Determinismus je Seed;
  **T → 0** nimmt keine Verschlechterung an und endet in einem 2-opt-Optimum, **T → ∞** nimmt alles an; **Boltzmann-Verteilung**: eine lange Kette bei festem T auf einer Instanz mit 6 Knoten besucht die 60 Touren im Verhältnis e^(−L/T) (Totalvariationsabstand unter 0.06, für Tausch, 2-opt, Or-opt und die Mischung) – das prüft Vorschlagssymmetrie, Metropolis-Regel und Längenänderung zugleich; Annahmequote gegen die Formel.
- Übernommener Kern: 2-opt gegen Brute-Force, Abstieg strikt monoton und im lokalen Optimum, Bewertungsbudget, 1-Baum-Schranke gegen Brute-Force (n = 8) und CP-SAT (n = 20); Instanz gegen eingefrorene Werte.
- **Alle Zahlen der App-Texte sind als Tests hinterlegt** (Seitenleiste, Presets, Grenzen-Tabelle, Budget-, Temperatur-, Plan-, Nachbarschafts-, Stufen-, Größen-, Streuungs- und Heatmap-Aussagen, Schranke und Optimum gegen CP-SAT; jeweils Mittel über die festen Sweep-Instanzen × Ketten; positive **und** negative Aussagen; Rechenzeiten nur als Größenordnung);
  alle 8 Presets über mehrere Instanzen und Ketten in Urteil-Bändern; AppTest-Rauchtests (Voreinstellung, jedes Preset, jeder Schritt bei 10 und 60 Stopps, Stufen-Regler, ▶️ Abspielen und ▶️ Abkühlen abspielen ohne doppelte Schlüssel, ausgeblendete Endtemperatur, Begrenzung, Würfel-Knöpfe, Permalink-Grenzen, Extremwerte, Experimente auf Abruf, Footer).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Schritte, Ergebnis, 📐 Sweeps, 🔬 Experimente (Budget, Temperaturfenster, Streuung, Skalierung), 🚧 Grenzen, Mathe |
| `sa_algorithm.py` | Metropolis-Kette, Abkühlpläne, Vorschläge (2-opt, Or-opt, Tausch) |
| `sa_tour.py` | Nachbarschaften, Abstieg (mit Bewertungsbudget), Kreuzungen, 1-Baum-Schranke (aus der Hill-Climbing-Demo) |
| `sa_scenario.py`, `sa_constants.py` | Instanzen (gleichverteilt, in Gruppen); Konstanten, Presets |
| `sa_evaluation.py` | Analyse, Urteil, Hill Climbing mit Neustarts, Sweeps, Temperaturfenster, Skalierung, Streuung |
| `sa_presets.py`, `sa_visualization.py` | Permalink/Presets (ausgeblendete Endtemperatur), Plotly-Figuren (achsengesperrt) |
| `tests/` | Kette (Boltzmann-Verteilung, Grenzfälle), übernommener Kern, Szenario und Auswertung, Aussagen der App, Presets, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
