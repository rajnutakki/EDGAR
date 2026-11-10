import jax.numpy as jnp

DT = 0.1  # integration step, ms (matches the data-generating grid)


def initial_state(params):
    TS = 2.0
    LTS = 0.0
    return {
        "LS": params["N"] - TS - LTS,
        "TS": TS,
        "LTS": LTS,
        "RP": 3.0,
        "Ca": 0.0,
    }


def model(hidden_state, y_prev, params):
    """Coupled-state response model with an auxiliary replenishment state.

    State variables:
        LS: intermediate state that supplies TS and LTS.
        TS: response-generating state depleted by stimulus events.
        LTS: additional response-generating state populated by stimulus events.
        RP: auxiliary state that supplies TS and relaxes toward a target level.
        Ca: stimulus-driven modulation signal with exponential decay.

    Inputs:
        hidden_state: dict with keys {"LS", "TS", "LTS", "RP", "Ca"}.
        y_prev: dict containing:
            stim_prev: binary stimulus indicator for the current step;
                P_prev is accepted as an alternative key.
            release_prev: previous model-predicted response (unused here).
        params: dict of free model parameters.

    Returns:
        Updated hidden state and the response amplitude for the current step.
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
    k2rest = params["k2rest"]
    k1rest = params["k1rest"]
    s1 = params["s1"]
    s2 = params["s2"]
    NRPoverN = params["NRPoverN"]

    # Functions
    N_RP = N * NRPoverN
    SumLS_TS_LTS = LS_prev + TS_prev + LTS_prev
    sigma2 = s2 * cadecay / deltaca
    sigma1 = s1 * cadecay / deltaca
    k2 = k2rest + sigma2 * Ca_prev
    k1 = (k1rest + (sigma1 * Ca_prev)) * ((N - SumLS_TS_LTS) / N)

    # Event response and stimulus-driven state transfer
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
    J8 = b3 * LTS_prev
    J9 = release2 / DT
    J10 = RP_prev * k2 * ((N - SumLS_TS_LTS) / N)
    J11 = k5 * (N_RP - RP_prev) / N_RP
    J6 = Ca_prev * cadecay
    J7 = P * deltaca / DT

    # Euler update
    LS = jnp.maximum(LS_prev + DT * (-J2 + J3 + J1 - J4 + J8), 0.0)
    TS = jnp.maximum(TS_prev + DT * (+J2 - J5 - J3 + J10), 0.0)
    LTS = LTS_prev + DT * (+J4 - J8 - J9)
    RP = RP_prev + DT * (-J10 + J11)
    Ca = jnp.maximum(Ca_prev + DT * (-J6 + J7), 0.0)

    new_hidden_state = {
        "LS": LS,
        "TS": TS,
        "LTS": LTS,
        "RP": RP,
        "Ca": Ca,
    }

    return new_hidden_state, release


model.DEFAULT_PARAMS = {
    "N": 10.0,
    "Pf": 0.7,
    "b2": 0.001,
    "b3": 0.01,
    "k": 0.3,
    "NRPoverN": 0.7,
    "k5": 0.005,
    "deltaca": 0.1,
    "cadecay": 0.01,
    "k2rest": 0.001,
    "k1rest": 0.008,
    "s1": 0.2,
    "s2": 0.1,
}
model.INITIAL_STATE = initial_state
