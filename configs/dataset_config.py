
META_CONFIGS = {
    "eeg": {
        "trlucb": {"kappa1": 2, "lmd" : 0.0000001, "m2": 1, "sigma_e": 1},
        "oful": {"lmd" : 0.1, "m2": 1, "sigma_e": 1},
        "linTS": {"delta": 0.99, "r": 0.1, "sigma_e": 1}
    },
    "cardiotocography": {
        "trlucb": {"kappa1": 1.3, "lmd" : 0.0000001, "m2": 1, "sigma_e": 1},
        "oful": {"lmd" : 0.1, "m2": 1, "sigma_e": 1},
        "linTS": {"delta": 0.99, "r": 0.1, "sigma_e": 1}
    },
    "eye_movements": {
        "trlucb": {"kappa1": 1.8, "lmd" : 0.0000001, "m2": 1, "sigma_e": 1},
        "oful": {"lmd" : 0.5, "m2": 1, "sigma_e": 1},
        "linTS": {"delta": 0.99, "r": 0.1, "sigma_e": 1}
    },
    "warfarin": {
        "trlucb": {"kappa1": 1.0, "lmd" : 0.0000001, "m2": 1, "sigma_e": 1},
        "oful": {"lmd" : 0.5, "m2": 1, "sigma_e": 1},
        "linTS": {"delta": 0.99, "r": 0.1, "sigma_e": 1}        
    }
}