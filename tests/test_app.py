"""AppTest-Rauchtests: Voreinstellung, jedes Preset, jeder Schritt, Randwerte, Abspielen ohne doppelte Schlüssel, ausgeblendete Endtemperatur, Experimente auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import sa_constants as C
import sa_evaluation as ev

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(sa_step=1, **state):
    at = AppTest.from_file(APP, default_timeout=300)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    if sa_step != 1:                                                       # die App setzt den Schritt beim ersten Lauf zurück: erst danach wählen
        at.select_slider(key="sa_step").set_value(sa_step).run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def _metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def test_default_run_has_no_exception_and_shows_the_measured_default():
    at = _run()
    _ok(at)
    assert _metric(at, "Beste Tour") == "1.3 %" and _metric(at, "Ein Hill-Climbing-Abstieg") == "6.8 %" and _metric(at, "Hill Climbing mit Neustarts") == "3.9 %"
    assert any("Besser als Hill Climbing" in s.value for s in at.success)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = C.PRESETS[name]
    assert at.session_state["schedule_select"] == p["schedule"] and at.session_state["budget_select"] == p["budget"] and at.session_state["n_slider"] == p["n"]
    assert at.metric


@pytest.mark.parametrize("step", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("n", [10, 60])
def test_every_step_runs(step, n):
    at = _run(n_slider=n, budget_select=25000, sa_step=step)
    _ok(at)
    assert at.get("plotly_chart") and at.session_state["sa_step"] == step


def test_step_four_level_slider_and_a_single_level():
    at = _run(sa_step=4, budget_select=25000)
    _ok(at)
    lv = next(s for s in at.slider if s.key == "sa_level")
    assert lv.value == lv.max == 100
    lv.set_value(3).run()
    _ok(at)
    at2 = _run(levels_select=1, budget_select=25000, sa_step=4)          # eine Stufe: kein Stufen-Regler
    _ok(at2)
    assert not [s for s in at2.slider if s.key == "sa_level"]


def test_play_runs_through_all_steps_and_the_cooling_without_duplicate_keys():
    at = _run(n_slider=20, budget_select=10000)
    next(b for b in at.button if b.label == "▶️ Abspielen").click().run()
    _ok(at)
    at2 = _run(n_slider=20, budget_select=10000, sa_step=4)
    next(b for b in at2.button if b.label == "▶️ Abkühlen abspielen").click().run()
    _ok(at2)


def test_end_temperature_is_hidden_for_the_logarithmic_plan_and_kept():
    at = _run(tend_slider=0.2, budget_select=25000)
    assert any(s.key == "tend_slider" for s in at.slider)
    at.selectbox(key="schedule_select").set_value("log").run()
    _ok(at)
    assert not any(s.key == "tend_slider" for s in at.slider)
    at.selectbox(key="schedule_select").set_value("geometric").run()
    _ok(at)
    assert next(s for s in at.slider if s.key == "tend_slider").value == pytest.approx(0.2)


def test_end_temperature_above_start_is_capped_with_a_warning():
    at = _run(t0_slider=0.1, tend_slider=0.5, budget_select=25000)
    _ok(at)
    assert any("begrenzt" in w.value for w in at.sidebar.warning)


def test_dice_buttons_change_the_seeds():
    at = _run(budget_select=10000)
    old = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neue Instanz generieren").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old
    old_c = at.session_state["chain_seed_input"]
    next(b for b in at.button if b.label == "🎲 Neue Kette würfeln").click().run()
    _ok(at)
    assert at.session_state["chain_seed_input"] != old_c


@pytest.mark.parametrize("kw", [dict(n_slider=200, budget_select=50000), dict(n_slider=10, ballung_slider=100, budget_select=10000), dict(neighborhood_select="swap", budget_select=25000),
                                dict(neighborhood_select="oropt", schedule_select="linear", n_slider=30, budget_select=25000), dict(start_radio="nearest", budget_select=25000), dict(levels_select=1000, budget_select=25000)])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))


def test_permalink_values_are_clamped_and_snapped():
    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params["n"] = "9999"
    at.query_params["ballung"] = "40"
    at.query_params["nb"] = "nonsense"
    at.query_params["t0"] = "0.52"
    at.query_params["budget"] = "12345"
    at.run()
    _ok(at)
    assert at.session_state["n_slider"] == C.N_MAX and at.session_state["ballung_slider"] == 50 and at.session_state["neighborhood_select"] == C.DEFAULT_NEIGHBORHOOD
    assert at.session_state["t0_slider"] == pytest.approx(0.5) and at.session_state["budget_select"] == C.DEFAULT_BUDGET


def test_sweeps_run_on_demand():
    at = _run(n_slider=10, budget_select=10000)
    at.selectbox(key="sweep_select").set_value("schedule").run()
    next(b for b in at.button if b.key == "sweep_start").click().run()
    _ok(at)
    assert at.get("plotly_chart")


def test_experiments_run_on_demand(monkeypatch):
    monkeypatch.setitem(ev.SWEEP_VALUES, "budget", (2000, 5000))
    monkeypatch.setattr(ev, "HEAT_BUDGET", 3000)
    monkeypatch.setattr(C, "SCALING_N", (10, 20))
    monkeypatch.setattr(ev, "SCALING_POLICIES", (("Budget 4 Tausend", lambda n: 4000), ("Budget 300 · Stopps", lambda n: 300 * n)))
    at = _run(n_slider=10, budget_select=10000)
    for key, flag in (("budget_start", "budget_on"), ("heatmap_start", "heatmap_on"), ("spread_start", "spread_on"), ("scaling_start", "scaling_on")):
        next(b for b in at.button if b.key == key).click().run()
        _ok(at)
        assert at.session_state[flag]


def test_budget_experiment_shows_dlb_column_only_for_two_opt(monkeypatch):
    monkeypatch.setitem(ev.SWEEP_VALUES, "budget", (2000, 5000))
    at = _run(n_slider=10, budget_select=10000)
    next(b for b in at.button if b.key == "budget_start").click().run()
    _ok(at)
    tbl = next(t for t in at.get("table") if "Budget" in t.value)
    assert "HC + Neustarts, Kandidatenliste + DLB (%)" in tbl.value
    at.selectbox(key="neighborhood_select").set_value("oropt").run()
    _ok(at)
    tbl2 = next(t for t in at.get("table") if "Budget" in t.value)
    assert "HC + Neustarts, Kandidatenliste + DLB (%)" not in tbl2.value


def test_footer_and_grenzen_are_present():
    at = _run(budget_select=10000)
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
    assert any("Die Temperatur passt zur Instanz" in m.value for m in at.markdown)
