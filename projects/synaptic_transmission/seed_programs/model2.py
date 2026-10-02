import numpy as np

DT = 0.1  # integration step, ms (matches the data-generating grid)


def model(hidden_state, y_prev, params):
    """Replacement Pool (RP) model of synaptic transmission.

    Reservoirs:
        - LS: synaptic vesicles in loosely docked state
        - TS: synaptic vesicles in tightly docked state
        - LTS: synaptic vesicles in labile tightly docked state
        - RP: synaptic vesicles in replacement pool
        - Ca: AP-induced effective [Ca2+] transient above basal rest (uM)

    Inputs:
        hidden_state: dict with keys {"LS", "TS", "LTS", "RP", "Ca"}
        y_prev: dict with keys:
            - "stim_prev" (or "P_prev"): action potential stimulus at t-1 (1 if pulse present, 0 otherwise)
            - "release_prev": previous observed release (optional)
        params: dict of learnable model parameters
    """
    LS_prev = hidden_state["LS"]
    TS_prev = hidden_state["TS"]
    LTS_prev = hidden_state["LTS"]
    RP_prev = hidden_state["RP"]
    Ca_prev = hidden_state["Ca"]

    P = y_prev.get("stim_prev", y_prev.get("P_prev", 0.0))

    N = params["N"]
    Pf = params["Pf"]
    b2 = params["b2"]
    b3 = params["b3"]
    k = params["k"]
    k5 = params["k5"]
    deltaca = params["deltaca"]
    cadecay = params["cadecay"]
    Crest = params["Crest"]
    k2rest = params["k2rest"]
    k1rest = params["k1rest"]
    s1 = params["s1"]
    s2 = params["s2"]
    NRPoverN = params["NRPoverN"]

    # Functions
    N_RP = N * NRPoverN
    SumLS_TS_LTS = LS_prev + TS_prev + LTS_prev
    totalCa = Ca_prev + Crest
    sigma2 = s2 / (cadecay * deltaca)
    sigma1 = s1 / (cadecay * deltaca)
    k2 = k2rest + sigma2 * (totalCa - Crest)
    k1 = (k1rest + (sigma1 * (totalCa - Crest))) * ((N - SumLS_TS_LTS) / N)

    # Evoked release and transition by stimulus
    release1 = Pf * TS_prev * P
    release2 = Pf * LTS_prev * P
    release = release1 + release2
    F1 = LS_prev * k * P

    # Flows
    J1 = k1
    J2 = LS_prev * k2
    J3 = b2 * TS_prev
    J4 = F1 / DT
    J5 = release1 / DT
    J8 = LTS_prev / b3
    J9 = release2 / DT
    J10 = RP_prev * k2 * ((N - SumLS_TS_LTS) / N)
    J11 = k5 * (N_RP - RP_prev) / N_RP
    J6 = Ca_prev / cadecay
    J7 = P * deltaca / DT

    # Euler update
    LS = np.maximum(LS_prev + DT * (-J2 + J3 + J1 - J4 + J8), 0.0)
    TS = np.maximum(TS_prev + DT * (+J2 - J5 - J3 + J10), 0.0)
    LTS = LTS_prev + DT * (+J4 - J8 - J9)
    RP = RP_prev + DT * (-J10 + J11)
    Ca = np.maximum(Ca_prev + DT * (-J6 + J7), 0.0)

    new_hidden_state = {
        "LS": LS,
        "TS": TS,
        "LTS": LTS,
        "RP": RP,
        "Ca": Ca,
    }

    return new_hidden_state, release


model.DEFAULT_PARAMS = {
    "N": 100.0,
    "Pf": 0.3,
    "b2": 0.001,
    "b3": 100.0,
    "k": 0.1,
    "NRPoverN": 2.0,
    "k5": 0.01,
    "deltaca": 1.0,
    "cadecay": 10.0,
    "Crest": 0.05,
    "k2rest": 0.001,
    "k1rest": 0.001,
    "s1": 0.1,
    "s2": 0.1,
    "s0_LS": 70.0,
    "s0_TS": 30.0,
    "s0_LTS": 0.0,
    "s0_RP": 200.0,
    "s0_Ca": 0.0,
    "log_noise_coef": -4.6052,
}
