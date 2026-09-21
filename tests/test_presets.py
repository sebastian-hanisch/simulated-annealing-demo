"""Presets: Vollständigkeit, gültige Werte, Urteile über mehrere Instanzen und Ketten (Bänder), Permalink-Konstanten."""

import pytest

import sa_constants as C
import sa_evaluation as ev
import sa_presets as P


def _settings(p, seed=None, chain_seed=None):
    return ev.Settings(n=p["n"], cluster_share=p["ballung"], seed=p["seed"] if seed is None else seed, neighborhood=p["neighborhood"], schedule=p["schedule"], t0=p["t0"], t_end=p["t_end"],
                       budget=p["budget"], levels=p["levels"], start=p["start"], chain_seed=p["chain_seed"] if chain_seed is None else chain_seed)


def test_every_preset_has_help_bands_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP) == set(C.PRESET_EXPECTED_BANDS) and len(C.PRESETS) == 8
    for name, p in C.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS) and C.PRESET_HELP[name]


def test_preset_values_are_valid_and_match_the_setting_specs():
    for name, p in C.PRESETS.items():
        assert C.N_MIN <= p["n"] <= C.N_MAX and (p["n"] - C.N_MIN) % C.N_STEP == 0
        assert C.BALLUNG_MIN <= p["ballung"] <= C.BALLUNG_MAX and p["ballung"] % C.BALLUNG_STEP == 0
        assert p["neighborhood"] in C.NEIGHBORHOOD_LABELS and p["schedule"] in C.SCHEDULE_LABELS and p["start"] in C.START_LABELS
        assert C.T0_MIN <= p["t0"] <= C.T0_MAX and C.T_END_MIN <= p["t_end"] <= C.T_END_MAX and p["t_end"] <= p["t0"]
        assert p["budget"] in C.BUDGETS and p["levels"] in C.LEVEL_OPTIONS
        for key, state_key in P.PRESET_KEYS.items():
            P.SETTING_SPECS[state_key].caster(p[key])


def test_default_preset_equals_the_default_settings():
    assert _settings(C.PRESETS["Standardfall (Voreinstellung)"]) == ev.Settings()
    assert P.KEPT == {"tend_slider": "_kept_tend_slider"}


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_preset_verdicts_stay_in_their_bands_over_instances_and_chains(name):
    p = C.PRESETS[name]
    seeds = range(2) if p["n"] >= 150 else range(5)
    seen = {ev.verdict(ev.analyse(_settings(p, seed=seed, chain_seed=ch), keep_snapshots=False)) for seed in seeds for ch in (0, 1)}
    assert seen <= C.PRESET_EXPECTED_BANDS[name], seen
    assert ev.verdict(ev.analyse(_settings(p), keep_snapshots=False)) in C.PRESET_EXPECTED_BANDS[name]


def test_preset_contrasts_at_the_default_instance():
    g = {name: ev.analyse(_settings(p), keep_snapshots=False) for name, p in C.PRESETS.items() if p["n"] == C.DEFAULT_N}
    std = g["Standardfall (Voreinstellung)"]
    assert g["Zu kalt (wie Hill Climbing)"].gap > std.gap + 0.5 or g["Zu kalt (wie Hill Climbing)"].worse_fraction < 0.001 < std.worse_fraction
    hot = g["Zu heiß am Ende"]
    assert hot.final_gap - hot.gap >= ev.HOT_END_GAP and hot.gap > std.gap
    assert g["Großes Budget (1 Million)"].run.proposals > std.run.proposals and g["Kleines Budget (25 Tausend)"].run.proposals < std.run.proposals
    assert g["Logarithmischer Plan"].gap > std.gap


def test_bounds_and_snapping_constants():
    assert P.bounds("n_slider") == (C.N_MIN, C.N_MAX) and P.bounds("seed_input") == (0, C.SEED_MAX) and P.bounds("t0_slider") == (C.T0_MIN, C.T0_MAX)
    assert P.STEPS == {"n_slider": C.N_STEP, "ballung_slider": C.BALLUNG_STEP} and P.FLOAT_STEPS == {"t0_slider": C.T0_STEP, "tend_slider": C.T_END_STEP}
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)
    assert C.DEFAULT_BUDGET in C.BUDGETS and C.DEFAULT_LEVELS in C.LEVEL_OPTIONS
