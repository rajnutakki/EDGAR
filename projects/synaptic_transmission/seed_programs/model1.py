import numpy as np

DT = 0.1  # integration step, ms (matches the data-generating grid)


def initial_state(params):
    """Return the fixed Berkeley Madonna initialization for the Neher model."""
    return {
        "ES": params["Ntotal"],
        "LS": 0.0,
        "TS": 0.0,
        "Ca": 0.0,
        "LTS": 0.0,
    }

def model(hidden_state, y_prev, params):
    """Neher model of synaptic transmission with multi-step vesicle priming and calcium dynamics.

    Reservoirs:
        - ES: empty release sites
        - LS: synaptic vesicles in loosely docked state
        - TS: synaptic vesicles in tightly docked state
        - LTS: synaptic vesicles in labile tightly docked state
        - Ca: AP-induced effective [Ca2+] transient above basal rest (uM)

    Inputs:
        hidden_state: dict with keys {"ES", "LS", "TS", "Ca", "LTS"}
        y_prev: dict with keys:
            - "stim_prev" (or "P_prev"): action potential stimulus at t-1 (1 if pulse present, 0 otherwise)
            - "release_prev": previous observed release (optional)
        params: dict of learnable model parameters
    """
    ES_prev = hidden_state["ES"]
    LS_prev = hidden_state["LS"]
    TS_prev = hidden_state["TS"]
    Ca_prev = hidden_state["Ca"]
    LTS_prev = hidden_state["LTS"]

    P = y_prev.get("stim_prev", y_prev.get("P_prev", 0.0))

    b1 = params["b1"]
    b2 = params["b2"]
    b3 = params["b3"]
    Pf = params["Pf"]
    k = params["k"]
    k_1 = params["k_1"]
    deltaca = params["deltaca"]
    cadecay = params["cadecay"]
    Crest = params["Crest"]
    k2rest = params["k2rest"]
    k1rest = params["k1rest"]
    s1 = params["s1"]
    s2 = params["s2"]

    # Calcium dynamics and transition rate constants
    totalCa = Ca_prev + Crest
    sigma2 = s2 / (cadecay * deltaca)
    sigma1 = s1 / (cadecay * deltaca)
    k2 = k2rest + sigma2 * (totalCa - Crest)
    k1 = (k1rest + sigma1 * (totalCa - Crest)) / (1.0 + (totalCa - Crest) / k_1)

    # Evoked release and transition by stimulus
    release1 = Pf * TS_prev * P
    release2 = Pf * LTS_prev * P
    release = release1 + release2
    F1 = k * LS_prev * P

    # Fluxes
    J1 = ES_prev * k1
    J2 = LS_prev * k2
    J3 = b2 * TS_prev
    J4 = LS_prev * b1
    J6 = Ca_prev / cadecay
    J7 = P * deltaca / DT
    J5 = release1 / DT
    J8 = F1 / DT
    J9 = LTS_prev / b3
    J10 = release2 / DT

    # Euler integration
    ES = np.maximum(ES_prev + DT * (-J1 + J4 + J5 + J10), 0.0)
    LS = np.maximum(LS_prev + DT * (+J1 - J2 + J3 - J4 - J8 + J9), 0.0)
    TS = np.maximum(TS_prev + DT * (+J2 - J3 - J5), 0.0)
    Ca = np.maximum(Ca_prev + DT * (-J6 + J7), 0.0)
    LTS = LTS_prev + DT * (+J8 - J9 - J10)

    new_hidden_state = {
        "ES": ES,
        "LS": LS,
        "TS": TS,
        "Ca": Ca,
        "LTS": LTS,
    }

    return new_hidden_state, release


model.DEFAULT_PARAMS = {
    "b1": 0.0001,
    "b2": 0.0002,
    "b3": 150.0,
    "Pf": 0.6,
    "k": 0.2,
    "k_1": 0.15,
    "deltaca": 0.1,
    "cadecay": 100.0,
    "Crest": 0.05,
    "k2rest": 0.0002,
    "k1rest": 0.0003,
    "s1": 0.3,
    "s2": 0.2,
    "Ntotal": 10.0,
}
model.INITIAL_STATE = initial_state
