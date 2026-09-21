"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Buttons (Standardmuster aus dem Demo-Portfolio, siehe bf_presets.py in bfs-demo)."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import sa_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


def _choice(options):
    def cast(value):
        value = str(value)
        if value not in options:
            raise ValueError(value)
        return value
    return cast


def _int_choice(options):
    def cast(value):
        value = int(value)
        if value not in options:
            raise ValueError(value)
        return value
    return cast


SETTING_SPECS = {
    "n_slider": SettingSpec("n", int, C.DEFAULT_N, C.N_MIN, C.N_MAX),
    "ballung_slider": SettingSpec("ballung", int, C.DEFAULT_BALLUNG, C.BALLUNG_MIN, C.BALLUNG_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
    "neighborhood_select": SettingSpec("nb", _choice(C.NEIGHBORHOOD_LABELS), C.DEFAULT_NEIGHBORHOOD),
    "schedule_select": SettingSpec("plan", _choice(C.SCHEDULE_LABELS), C.DEFAULT_SCHEDULE),
    "t0_slider": SettingSpec("t0", float, C.DEFAULT_T0, C.T0_MIN, C.T0_MAX),
    "tend_slider": SettingSpec("tend", float, C.DEFAULT_T_END, C.T_END_MIN, C.T_END_MAX),
    "budget_select": SettingSpec("budget", _int_choice(C.BUDGETS), C.DEFAULT_BUDGET),
    "levels_select": SettingSpec("levels", _int_choice(C.LEVEL_OPTIONS), C.DEFAULT_LEVELS),
    "start_radio": SettingSpec("start", _choice(C.START_LABELS), C.DEFAULT_START),
    "chain_seed_input": SettingSpec("cseed", int, C.DEFAULT_CHAIN_SEED, 0, C.SEED_MAX),
}
PRESET_KEYS = {"n": "n_slider", "ballung": "ballung_slider", "seed": "seed_input", "neighborhood": "neighborhood_select", "schedule": "schedule_select", "t0": "t0_slider", "t_end": "tend_slider",
               "budget": "budget_select", "levels": "levels_select", "start": "start_radio", "chain_seed": "chain_seed_input"}
# Regler, die bei einem Abkühlplan ohne Endtemperatur (logarithmisch) ausgeblendet sind: Streamlit löscht ihren Zustand, sobald sie nicht gezeichnet werden - der zuletzt gewählte Wert bleibt hier erhalten
KEPT = {"tend_slider": "_kept_tend_slider"}
STEPS = {"n_slider": C.N_STEP, "ballung_slider": C.BALLUNG_STEP}
FLOAT_STEPS = {"t0_slider": C.T0_STEP, "tend_slider": C.T_END_STEP}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = st.session_state.get(KEPT[state_key], spec.default) if state_key in KEPT else spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    for key, step in STEPS.items():
        if key in st.session_state:
            lo = SETTING_SPECS[key].lo
            st.session_state[key] = int(lo + round((st.session_state[key] - lo) / step) * step)
    for key, step in FLOAT_STEPS.items():
        if key in st.session_state:
            lo = SETTING_SPECS[key].lo
            st.session_state[key] = round(lo + round((st.session_state[key] - lo) / step) * step, 4)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = C.PRESETS[name][key]
        if state_key in KEPT:
            st.session_state[KEPT[state_key]] = C.PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)


def randomize_chain_seed():
    st.session_state["chain_seed_input"] = random.randint(0, C.SEED_MAX)
