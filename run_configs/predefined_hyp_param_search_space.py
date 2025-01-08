from itertools import product

from arfs.feature_selection.lasso import LassoFeatureSelection
from ray import tune
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, mutual_info_classif
from sklearn.metrics import get_scorer
from umap import UMAP


def get_rf_search_space(best_fit_scorer: str):
    rf_search_space = {
        "model__n_estimators": tune.randint(10, 500),  # Replace get_rf_search_space logic
        "model__max_depth": tune.choice([None, tune.randint(10, 200)]),
        "model__criterion": tune.choice(["gini", "entropy"]),
        "model__class_weight": tune.choice(["balanced", None]),
        "model__oob_score": tune.choice([False, get_scorer(best_fit_scorer)._score_func]),
    }

    return rf_search_space


def get_lgbm_search_space():
    # "model__lgmb__boosting_type" : ["gbdt", "dart"], # TODO
    # "model__lgmb__n_estimators": [20, 50, 100],
    # "model__lgmb__learning_rate": [.1, .01, .001, .05, .5],
    # "model__lgmb__num_leaves": [31, 50, 100, 1000, 10000, 100000],
    # "model__lgmb__colsample_bytree": [0.4, 0.7, 1],
    # "model__lgmb__subsample": [0, 1, 10, 25],
    # "model__lgmb__num_iterations": [100, 1000, 10000],
    # "model__lgmb__max_depth": [-1, 10, 100],
    # "model__lgmb__min_child_samples": [20, 10, 50],
    # "model__lgmb__reg_alpha": [.0, ],
    # "model__lgmb__reg_lambda": [.0, ],
    # ...

    # "model__lgmb__subsample": ["bagging"]
    ...


umap_search_space = UMAP(
    n_components=tune.choice([30, 70]),
    n_neighbors=tune.choice([50, 100]),
    min_dist=tune.uniform(0.25, 0.8),
    metric=tune.choice(["braycurtis", "canberra"]),
)

select_percentile_search_space = SelectPercentile(
    mutual_info_classif(
        discrete_features=False, n_neighbors=tune.randint([2, 5, 10, 20])
    ),
    percentile=tune.choice([10, 50, 80]),
)
