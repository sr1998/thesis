import run_configs.optuna_search_space_samplers as sss

def get_setup(algorithm: str):
    search_space_sampler = {
        "MAML": sss.maml_search_space_sampler,
        "Reptile": sss.reptile_search_space_sampler,
        "ProtoNet": sss.protonet_search_space_sampler,
    }[algorithm]
    
    n_outer_splits = 10
    n_inner_splits = 3
    tuning_mode = "maximize"
    best_fit_scorer = "f1"
    n_parallel_jobs = 5
    tuning_num_samples_helper = 20
    tuning_num_samples_primary = tuning_num_samples_helper + 5 # 5 is needed for warmp-up
    tuning_num_samples = tuning_num_samples_helper * n_parallel_jobs + 5


    return {
        "n_outer_splits": n_outer_splits,
        "n_inner_splits": n_inner_splits,
        "tuning_mode": tuning_mode,
        "best_fit_scorer": best_fit_scorer,
        "tuning_num_samples": tuning_num_samples,
        "tuning_num_samples_primary": tuning_num_samples_primary,
        "tuning_num_samples_helper": tuning_num_samples_helper,
        "search_space_sampler": search_space_sampler,
        "initial_trial": sss.PROTONET_INTIAL_TRIAL_FOR_OPTUNA if algorithm == "ProtoNet" else None #sss.MAML_INTITIAL_TRIAL,
    }