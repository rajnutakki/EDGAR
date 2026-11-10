
def parameter_estimator(data):
    """RP model parameter estimator based on previous optimizations, in future use statistical methods to estimate from data"""
    return {
        "N": 10.0,
        "Pf": 0.69507,
        "b2": 0.00113,
        "b3": 0.00848,
        "k": 0.32232,
        "NRPoverN": 0.8,
        "k5": 0.0055,
        "deltaca": 0.11,
        "cadecay": 0.01226,
        "Crest": 0.05,
        "k2rest": 0.0001,
        "k1rest": 0.00795,
        "s1": 0.20331,
        "s2": 0.09100,
    }
