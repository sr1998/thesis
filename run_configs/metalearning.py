import run_configs.optuna_search_space_samplers as sss

def get_setup(algorithm: str):
    search_space_sampler = {
        "MAML": sss.maml_search_space_sampler,
        "Reptile": sss.reptile_search_space_sampler,
    }[algorithm]
    
    n_outer_splits = 10
    n_inner_splits = 3
    tuning_mode = "maximize"
    best_fit_scorer = "f1"
    tuning_num_samples = 25

    return {
        "n_outer_splits": n_outer_splits,
        "n_inner_splits": n_inner_splits,
        "tuning_mode": tuning_mode,
        "best_fit_scorer": best_fit_scorer,
        "tuning_num_samples": tuning_num_samples,
        "search_space_sampler": search_space_sampler,
    }