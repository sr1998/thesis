from lightgbm import LGBMClassifier
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, mutual_info_classif
from sklearn.metrics import average_precision_score, get_scorer, make_scorer, roc_auc_score
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import FunctionTransformer
from sklearn.preprocessing import Normalizer
from arfs.feature_selection.lasso import LassoFeatureSelection
from umap import UMAP

from src.helper_function import create_pipeline
from src.preprocessing.functions import NumpyReplace, total_sum_scaling
from src.preprocessing.micronorm import GMPR_normalize


def get_setup():
    # TODO Define the misc config. Here is a dummy example
    misc_config = {
        "wandb": True,  # whether to use wandb or not
        "wandb_params": {
            "project": "thesis_baselines",
            "group": "RF",  # model name can be useful here
        },
        "verbose_pipeline": True,  # whether to print verbose output from the pipeline
        "cache_pipeline_steps": True,
    }

    # outer_cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    outer_cv_config = {
        "type": ShuffleSplit,
        "params": {"n_splits": 5, "test_size": 0.2, "random_state": 42},
    }

    inner_cv_config = {
        "type": ShuffleSplit,
        "params": {
            "n_splits": 5,
            "test_size": 0.2,
        },  # don't provide random_state, as we want to change it per outer fold
    }

    preprocessor_pipeline = create_pipeline(
        [
            (
                "batch_effect_corrections_and_imputations",
                "passthrough",
            ),  # Together as imputation may be needed after/before batch effect correction depending on whether can handle sparse data or not
            ("normalizations_and_transformations", "passthrough"),
            ("feature_space_change", "passthrough"),
        ],
        misc_config,
    )

    label_preprocessor = LabelEncoder()

    standard_pipeline = create_pipeline(
        [
            ("preprocessor", preprocessor_pipeline),
            ("model", "passthrough"),
        ],
        misc_config,
    )

    score_functions = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1(_binary)": "f1",
        "f1_macro": "f1_macro",
        "f1_micro": "f1_micro",
        "f1_weighted": "f1_weighted",
        "roc_auc(_macro)": "roc_auc",
        "roc_auc_micro": make_scorer(roc_auc_score, average="micro"),
        "roc_auc_weighted": make_scorer(roc_auc_score, average="weighted"),
        "average_precision(_macro)": make_scorer(average_precision_score),
        "average_precision_micro": make_scorer(
            average_precision_score, average="micro"
        ),
        "average_precision_weighted": make_scorer(
            average_precision_score, average="weighted"
        ),
        "precision(_binary)": "precision",
        "precision_macro": "precision_macro",
        "precision_micro": "precision_micro",
        "precision_weighted": "precision_weighted",
        "recall(_binary)": "recall",
        "recall_micro": "recall_micro",
        "recall_macro": "recall_macro",
        "recall_weighted": "recall_weighted",
    }
    best_fit_scorer = "f1_weighted"

    # TODO Define the tuning grid. Here is a dummy example
    tuning_grid = [
        {
            "batch_effect_corrections_and_imputations": [
                NumpyReplace(),
                "passthrough",
            ],
            "preprocessor__normalizations_and_transformations": [
                Normalizer(norm="l1"),
                Normalizer(norm="l2"),
                FunctionTransformer(total_sum_scaling, validate=True, feature_names_out="one-to-one"),
                "passthrough",
            ],
            "preprocessor__feature_space_change": [
                LassoFeatureSelection(n_jobs=1),
                UMAP(),
                SelectPercentile(mutual_info_classif(discrete_features=False, n_neighbors=2)),
                SelectPercentile(mutual_info_classif(discrete_features=False, n_neighbors=5)),
                SelectPercentile(mutual_info_classif(discrete_features=False, n_neighbors=10)),
                SelectPercentile(mutual_info_classif(discrete_features=False, n_neighbors=20)),
                "passthrough"
            ],
            "preprocessor__feature_space_change__n_components": [30, 70],
            "preprocessor__feature_space_change__n_neighbors": [50, 100],
            "preprocessor__feature_space_change__min_dist": [0.25, 0.8],
            "preprocessor__feature_space_change__metric": ["braycurtis", "canberra"],
            "preprocessor__feature_space_change__percentile": [10, 50],
            "model": [RandomForestClassifier()],
            "model__n_estimators": [20, 50, 100],
            "model__max_depth": [None, 50, 100],
            "model__class_weight": ["balanced", None],
            "model__oob_score": [False, get_scorer(best_fit_scorer)._score_func],
            "model__criterion": ["gini", "entropy"],

        },
        {
            "batch_effect_corrections_and_imputations": ["passthrough"],
            "preprocessor__normalizations_and_transformations": [
                FunctionTransformer(GMPR_normalize, validate=True, feature_names_out="one-to-one"),
            ],
            "preprocessor__feature_space_change": [
                LassoFeatureSelection(n_jobs=1),
                UMAP(),
                SelectPercentile(mutual_info_classif(discrete_features=False, n_neighbors=2)),
                SelectPercentile(mutual_info_classif(discrete_features=False, n_neighbors=5)),
                SelectPercentile(mutual_info_classif(discrete_features=False, n_neighbors=10)),
                SelectPercentile(mutual_info_classif(discrete_features=False, n_neighbors=20)),
                "passthrough"
            ],
            "preprocessor__feature_space_change__n_components": [30, 70],
            "preprocessor__feature_space_change__n_neighbors": [50, 100],
            "preprocessor__feature_space_change__min_dist": [0.25, 0.8],
            "preprocessor__feature_space_change__metric": ["braycurtis", "canberra"],
            "preprocessor__feature_space_change__percentile": [10, 50],
            "model": [RandomForestClassifier()]
                # create_pipeline([("lgmb", LGBMClassifier())], misc_config),
            "model__rf__n_estimators": [20, 50, 100],
            "model__rf__max_depth": [None, 50, 100],
            "model__rf__class_weight": ["balanced", None],
            "model__rf__oob_score": [False, get_scorer(best_fit_scorer)._score_func],
            "model__rf__criterion": ["gini", "entropy"],
            # "model__lgmb__boosting_type" : ["gbdt", "dart"], # TODO
            # "model__lgmb__n_estimators": [20, 50, 100],
            # "model__lgmb__learning_rate": [.1, .01, .001, .05, .5],
            # "model__lgmb__num_leaves": [31, 50, 100, 1000, 10000, 100000],
            # "model__lgmb__colsample_bytree": [0.4, 0.7, 1],
            # "model__lgmb__subsample": [0, 1, 10, 25],
            # "model__lgmb__bagging_fraction": [.2, .5, .8, 1],
            # "model__lgmb__num_iterations": [100, 1000, 10000],
            # "model__lgmb__max_depth": [-1, 10, 100],
            # "model__lgmb__min_child_samples": [20, 10, 50],
            # "model__lgmb__reg_alpha": [.0, ],
            # "model__lgmb__reg_lambda": [.0, ],
            # ...

            # "model__lgmb__subsample": ["bagging"]
            
            
        },
    ]


    # tuner = GridSearchCV(
    #     estimator=standard_pipeline,
    #     param_grid=tuning_grid,
    #     cv=inner_cv_config,
    #     n_jobs=-1,
    #     verbose=2 if misc_config.get("verbose_pipeline", True) else 0,
    #     scoring=score_functions,
    #     refit=best_fit_scorer,  # This should be a string indicating the metric to use for refitting. We want refitting in any case.
    #     return_train_score=True,
    # )

    return {
        "misc_config": misc_config,
        "outer_cv_config": outer_cv_config,
        "inner_cv_config": inner_cv_config,
        "standard_pipeline": standard_pipeline,
        "label_preprocessor": label_preprocessor,
        "scoring": score_functions,
        "best_fit_scorer": best_fit_scorer,
        "tuning_grid": tuning_grid,
    }
