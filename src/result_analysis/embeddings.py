import os

import numpy as np
from src.scoring.metalearning_scoring_fn import compute_metrics
from visualizations.helper_functions import get_desired_cross_val_results_for_models
import torch
import optuna
import yaml

import fire
from run_configs.optuna_search_space_samplers import protonet_search_space_sampler
from src.helper_function import load_checkpoint
from src.models.metalearning_helpers import get_metalearning_model_from_trial
import pandas as pd
import torch.nn.functional as F
import wandb

# STUDIES=(
#     'ChenB_2020' 'YeZ_2018' 'ChuY_2021' 'ZhouC_2020' 'YeohYK_2021'
#     'HeQ_2017' 'HuY_2019' 'HuangR_2020' 'LiJ_2017' 'LiR_2021'
#     'LiuP_2021' 'LiuR_2017' 'LuW_2018' 'MaoL_2021'
#     'QiX_2019' 'QianY_2020' 'QinN_2014' 'WanY_2021'
#     'WangM_2019' 'WangX_2020' 'WengY_2019' 'YanQ_2017'
#     'YangY_2021' 'YeZ_2020' 'YuJ_2017'
#     'ZhangX_2015' 'ZhongH_2019' 'ZhuF_2020'
#     'ZhuJ_2018' 'ZhuQ_2021' 'ZuoK_2019'
#     'JieZ_2017' 'WangQ_2021' 'ZengQ_2021' 'HanL_2021'
#     'QinJ_2012'
# )

studies = ["ChenB_2020", "YeZ_2018", "ChuY_2021", "ZhouC_2020", "YeohYK_2021",
           "HeQ_2017", "HuY_2019", "HuangR_2020", "LiJ_2017", "LiR_2021",
           "LiuP_2021", "LiuR_2017", "LuW_2018", "MaoL_2021",
           "QiX_2019", "QianY_2020", "QinN_2014", "WanY_2021",
           "WangM_2019", "WangX_2020", "WengY_2019", "YanQ_2017",
           "YangY_2021", "YeZ_2020", "YuJ_2017",
           "ZhangX_2015", "ZhongH_2019", "ZhuF_2020",
           "ZhuJ_2018", "ZhuQ_2021", "ZuoK_2019",
           "JieZ_2017", "WangQ_2021", "ZengQ_2021", "HanL_2021",
           "QinJ_2012"]

def get_embeddings(study):
    abundance_data = pd.read_csv('/tudelft.net/staff-umbrella/abeellabstudents/sramezani/thesis/data/sun_et_al_data/mpa4_species_profile_preprocessed.csv', index_col=0, header=0)
    metadata = pd.read_csv("/tudelft.net/staff-umbrella/abeellabstudents/sramezani/thesis/data/sun_et_al_data/sample_group_species_preprocessed.csv", index_col=0, header=0)
    metadata = metadata.loc[abundance_data.index]

    study_names: list = metadata["Project_1"].unique().tolist()
    test_metadata = metadata[metadata["Project_1"] == study]
    test_data = abundance_data.loc[test_metadata.index]
    train_metadata = metadata.drop(index=test_metadata.index)
    train_data = abundance_data.drop(index=test_data.index)


    resume_path = "/tudelft.net/staff-umbrella/abeellabstudents/sramezani/thesis/checkpoints/metalearning2/ProtoNet/{0}/TS{0}_TK10_unbalanced_sun et al_ProtoNet_Tspecies_"
    data_splits = "/tudelft.net/staff-umbrella/abeellabstudents/sramezani/thesis/data/sun_et_al_data/selected_data_splits/{0}/10shot_nOuter10_nInner5_{0}_69189786.yml"

    resume_path = resume_path.format(study)
    data_splits = data_splits.format(study)

    file_path = f"/tudelft.net/staff-umbrella/abeellabstudents/sramezani/thesis/res_analysis/{study}"
    # create directory if not exists
    os.makedirs(file_path, exist_ok=True)


    # get test loop data split
    with open(data_splits, "r") as f:
        cross_val_data_selection = yaml.safe_load(f)
        test_loop_data_selection = cross_val_data_selection[0]

    checkpoint = load_checkpoint(str(resume_path + "/checkpoint.yaml"), True)

    storage_path = f"sqlite:///{resume_path}/optuna_study.db"
    storage = optuna.storages.RDBStorage(
        url=storage_path,
        heartbeat_interval=60,
        grace_period=120,
        failed_trial_callback=optuna.storages.RetryFailedTrialCallback(max_retry=1),
    )
    optuna_study = optuna.create_study(
        direction="maximize",
        study_name=f"hyper-param_optimization_for_{checkpoint['wandb_run_id']}",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
    )

    best_trial = optuna_study.best_trial
    best_trial_params = best_trial.params

    config = {
        "datasource": "sun et al",
        "algorithm": "ProtoNet",
        "balanced_or_unbalanced": "unbalanced",
        "resume": True,
        "positive_class_label": "Disease",
        "track_best_f1": True,
        "splitting_method": "study_wise",
        "feature_reduction_alg": "",
        "device": "cuda",
    }


    support_embeddings = {}
    support_labels = {}
    query_embeddings = {}
    query_labels = {}

    for i, test_support_set in test_loop_data_selection.items():
        best_trial_config = protonet_search_space_sampler(best_trial)
        best_model, train_loader, val_loader, test_loader = get_metalearning_model_from_trial(
            train_data,
            test_data,
            train_metadata,
            test_metadata,
            10,
            10,
            test_support_set,
            "ProtoNet",
            best_trial_config,
            config,
        )

        train_res, test_res = best_model.fit(
            train_dataloader=train_loader,
            n_epochs=200,
            n_parallel_tasks=5,
            val_loader=val_loader,
            eval_dataloader=test_loader,
            val_or_test="test",
            log_metrics=False,
            log_gradients=False,
            score_name_prefix=f"outer_fold_{i}_fit",
            save_best_model_path=None,
            track_best_f1=False,
            early_stopping_patience=best_trial_config["early_stopping_patience"],
            early_stopping_fraction=best_trial_config["early_stopping_fraction"],
            # early_stopping_metric=early_stop_metric
        )

        with torch.no_grad():
            for j, (X, y) in enumerate(test_loader):
                X, y = X.to(best_model.device), y.to(best_model.device, dtype=torch.int64)
                X_support = X[: best_model.eval_k_shot * 2, :]
                y_support = y[: best_model.eval_k_shot * 2]
                X_query = X[best_model.eval_k_shot * 2 :, :]
                y_query = y[best_model.eval_k_shot * 2 :]

                support_embedding = best_model.protonet.encoder(X_support)
                query_embedding = best_model.protonet.encoder(X_query)

                support_embeddings[i] = support_embedding.cpu().numpy().tolist()
                support_labels[i] = y_support.cpu().numpy().tolist()
                query_embeddings[i] = query_embedding.cpu().numpy().tolist()
                query_labels[i] = y_query.cpu().numpy().tolist()

        # save model state
        torch.save(
            best_model.protonet.state_dict(),
            os.path.join(file_path, f"model_fold_{i}.pth")
        )

    all_embeddings = {
        "support_embeddings": support_embeddings,
        "support_labels": support_labels,
        "query_embeddings": query_embeddings,
        "query_labels": query_labels,
    }

    # Save the embeddings and labels
    with open(file_path + f"embeddings_{study}.yml", "w") as f:
        yaml.dump(all_embeddings, f)
    print(f"Embeddings and labels for {study} saved to {file_path}")
    print(f"Finished processing study: {study}")


def get_best_embeddings(study):
    # get 
    metalearning_res = get_desired_cross_val_results_for_models(
        "metalearning2",
        ["ProtoNet"],
        ["unbalanced", "sun et al", "10_shot"],
        "Outer fold.test/f1",
        ["LuW_2018"],  # Use all studies
        notes_contains="'extra_str_indicator': ''"
        )[0]
    metalearning_res.columns = list(map(lambda x: "_".join(x.split("_")[:2])[2:], metalearning_res.columns))
    study_scores = metalearning_res.loc[:, study].dropna()
    study_scores = study_scores[study_scores != 0]
    median = np.median(study_scores.values)
    chosen_fold = min(study_scores.index, key=lambda k: abs(study_scores[k] - median))

    abundance_data = pd.read_csv('/tudelft.net/staff-umbrella/abeellabstudents/sramezani/thesis/data/sun_et_al_data/mpa4_species_profile_preprocessed.csv', index_col=0, header=0)
    metadata = pd.read_csv("/tudelft.net/staff-umbrella/abeellabstudents/sramezani/thesis/data/sun_et_al_data/sample_group_species_preprocessed.csv", index_col=0, header=0)
    metadata = metadata.loc[abundance_data.index]

    study_names: list = metadata["Project_1"].unique().tolist()
    test_metadata = metadata[metadata["Project_1"] == study]
    test_data = abundance_data.loc[test_metadata.index]
    train_metadata = metadata.drop(index=test_metadata.index)
    train_data = abundance_data.drop(index=test_data.index)


    resume_path = "/tudelft.net/staff-umbrella/abeellabstudents/sramezani/thesis/checkpoints/metalearning2/ProtoNet/{0}/TS{0}_TK10_unbalanced_sun et al_ProtoNet_Tspecies_"
    data_splits = "/tudelft.net/staff-umbrella/abeellabstudents/sramezani/thesis/data/sun_et_al_data/selected_data_splits/{0}/10shot_nOuter10_nInner5_{0}_69189786.yml"

    resume_path = resume_path.format(study)
    data_splits = data_splits.format(study)

    file_path = f"/tudelft.net/staff-umbrella/abeellabstudents/sramezani/thesis/res_analysis_median_fold/{study}"
    # create directory if not exists
    os.makedirs(file_path, exist_ok=True)


    # get test loop data split
    with open(data_splits, "r") as f:
        cross_val_data_selection = yaml.safe_load(f)
        test_loop_data_selection = cross_val_data_selection[0]

    checkpoint = load_checkpoint(str(resume_path + "/checkpoint.yaml"), True)

    storage_path = f"sqlite:///{resume_path}/optuna_study.db"
    storage = optuna.storages.RDBStorage(
        url=storage_path,
        heartbeat_interval=60,
        grace_period=120,
        failed_trial_callback=optuna.storages.RetryFailedTrialCallback(max_retry=1),
    )
    optuna_study = optuna.create_study(
        direction="maximize",
        study_name=f"hyper-param_optimization_for_{checkpoint['wandb_run_id']}",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
    )

    best_trial = optuna_study.best_trial
    best_trial_params = best_trial.params

    config = {
        "datasource": "sun et al",
        "algorithm": "ProtoNet",
        "balanced_or_unbalanced": "unbalanced",
        "resume": True,
        "positive_class_label": "Disease",
        "track_best_f1": True,
        "splitting_method": "study_wise",
        "feature_reduction_alg": "",
        "device": "cuda",
    }


    test_support_set = test_loop_data_selection[chosen_fold]
    # for i, test_support_set in test_loop_data_selection.items():
    best_trial_config = protonet_search_space_sampler(best_trial)
    best_model, train_loader, val_loader, test_loader = get_metalearning_model_from_trial(
        train_data,
        test_data,
        train_metadata,
        test_metadata,
        10,
        10,
        test_support_set,
        "ProtoNet",
        best_trial_config,
        config,
    )

    train_res, test_res = best_model.fit(
        train_dataloader=train_loader,
        n_epochs=200,
        n_parallel_tasks=5,
        val_loader=val_loader,
        eval_dataloader=test_loader,
        val_or_test="test",
        log_metrics=False,
        log_gradients=False,
        score_name_prefix=f"outer_fold_{chosen_fold}_fit",
        save_best_model_path=None,
        track_best_f1=False,
        early_stopping_patience=best_trial_config["early_stopping_patience"],
        early_stopping_fraction=best_trial_config["early_stopping_fraction"],
        # early_stopping_metric=early_stop_metric
    )

    with torch.no_grad():
        for j, (X, y) in enumerate(test_loader):
            X, y = X.to(best_model.device), y.to(best_model.device, dtype=torch.int64)
            X_support = X[: best_model.eval_k_shot * 2, :]
            y_support = y[: best_model.eval_k_shot * 2]
            X_query = X[best_model.eval_k_shot * 2 :, :]
            y_query = y[best_model.eval_k_shot * 2 :]

            support_embedding = best_model.protonet.encoder(X_support)
            query_embedding = best_model.protonet.encoder(X_query)

            support_embeddings = support_embedding.cpu().numpy().tolist()
            support_labels = y_support.cpu().numpy().tolist()
            query_embeddings = query_embedding.cpu().numpy().tolist()
            query_labels = y_query.cpu().numpy().tolist()
            original_support_data = X_support.cpu().numpy().tolist()
            original_support_labels = y_support.cpu().numpy().tolist()

    # save model state
    torch.save(
        best_model.protonet.state_dict(),
        os.path.join(file_path, f"model_fold_{chosen_fold}.pth")
    )

    all_embeddings = {
        "support_embeddings": support_embeddings,
        "support_labels": support_labels,
        "query_embeddings": query_embeddings,
        "query_labels": query_labels,
        "original_support_data": original_support_data,
        "original_support_labels": original_support_labels,
    }

    # Save the embeddings and labels
    with open(file_path + f"embeddings_{study}.yml", "w") as f:
        yaml.dump(all_embeddings, f)
    print(f"Data, Embeddings and labels for {study} saved to {file_path}")
    print(f"Finished processing study: {study}")


def best_model_predictions(study):
    abundance_data = pd.read_csv('/tudelft.net/staff-umbrella/abeellabstudents/sramezani/thesis/data/sun_et_al_data/mpa4_species_profile_preprocessed.csv', index_col=0, header=0)
    metadata = pd.read_csv("/tudelft.net/staff-umbrella/abeellabstudents/sramezani/thesis/data/sun_et_al_data/sample_group_species_preprocessed.csv", index_col=0, header=0)

    study_names: list = metadata["Project_1"].unique().tolist()
    test_metadata = metadata[metadata["Project_1"] == study]
    test_data = abundance_data.loc[test_metadata.index]
    train_metadata = metadata.drop(index=test_metadata.index)
    train_data = abundance_data.drop(index=test_data.index)

    resume_path = "/tudelft.net/staff-umbrella/abeellabstudents/sramezani/thesis/checkpoints/metalearning2/ProtoNet/{0}/TS{0}_TK10_unbalanced_sun et al_ProtoNet_Tspecies_"
    data_splits = "/tudelft.net/staff-umbrella/abeellabstudents/sramezani/thesis/data/sun_et_al_data/selected_data_splits/{0}/10shot_nOuter10_nInner5_{0}_69189786.yml"

    resume_path = resume_path.format(study)
    data_splits = data_splits.format(study)
    
    file_path = f"/tudelft.net/staff-umbrella/abeellabstudents/sramezani/thesis/res_analysis/{study}"
    metalearning_res = get_desired_cross_val_results_for_models(
        "metalearning2",
        ["ProtoNet"],
        ["unbalanced", "sun et al", "10_shot"],
        "Outer fold.test/f1",
        ["LuW_2018"],  # Use all studies
        notes_contains="'extra_str_indicator': ''"
        )[0]
    metalearning_res.columns = list(map(lambda x: "_".join(x.split("_")[:2])[2:], metalearning_res.columns))


    study_scores = metalearning_res.loc[:, study].dropna()
    study_scores = study_scores[study_scores != 0]
    median = np.median(study_scores.values)
    chosen_fold = min(study_scores.index, key=lambda k: abs(study_scores[k] - median))


    with open(data_splits, "r") as f:
        cross_val_data_selection = yaml.safe_load(f)
        test_loop_data_selection = cross_val_data_selection[0]

    checkpoint = load_checkpoint(str(resume_path + "/checkpoint.yaml"), True)

    storage_path = f"sqlite:///{resume_path}/optuna_study.db"
    storage = optuna.storages.RDBStorage(
        url=storage_path,
        heartbeat_interval=60,
        grace_period=120,
        failed_trial_callback=optuna.storages.RetryFailedTrialCallback(max_retry=1),
    )
    optuna_study = optuna.create_study(
        direction="maximize",
        study_name=f"hyper-param_optimization_for_{checkpoint['wandb_run_id']}",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
    )

    best_trial = optuna_study.best_trial
    best_trial_params = best_trial.params

    config = {
        "datasource": "sun et al",
        "algorithm": "ProtoNet",
        "balanced_or_unbalanced": "unbalanced",
        "resume": True,
        "positive_class_label": "Disease",
        "track_best_f1": True,
        "splitting_method": "study_wise",
        "feature_reduction_alg": "",
        "device": "cpu",
    }

    best_trial_config = protonet_search_space_sampler(best_trial)
    model, train_loader, val_loader, test_loader, class_weights = get_metalearning_model_from_trial(
        train_data,
        test_data,
        train_metadata,
        test_metadata,
        10,
        10,
        test_loop_data_selection[chosen_fold],
        "ProtoNet",
        best_trial_config,
        config,
    )

    best_fitted_model_dict = torch.load(
        os.path.join(file_path, f"model_fold_{chosen_fold}.pth"),
        map_location=torch.device("cpu")
    )
    model.protonet.load_state_dict(best_fitted_model_dict)
    model.protonet.eval()

    predictions = []
    with torch.no_grad():
        for j, (X, y) in enumerate(test_loader):
            X, y = X.to(model.device), y.to(model.device, dtype=torch.int64)
            X_support = X[: model.eval_k_shot * 2, :]
            y_support = y[: model.eval_k_shot * 2]
            X_query = X[model.eval_k_shot * 2 :, :]
            y_query = y[model.eval_k_shot * 2 :]

            loss, y_hat, target_inds = model.protonet.loss(X_support, X_query, y_support, y_query, class_weights)
            predictions.append(y_support.cpu().numpy().tolist() + y_hat.cpu().numpy().tolist())

    # save predictions yml
    save_path = f"/tudelft.net/staff-umbrella/abeellabstudents/sramezani/thesis/res_analysis/predictions/"
    os.makedirs(save_path, exist_ok=True)
    with open(save_path + f"/predictions_{study}.yml", "w") as f:
        yaml.dump(predictions, f)
    print(f"Predictions for {study} saved")


    # Here you can implement the logic to use the embeddings and labels for predictions
    # For example, you can use a classifier to predict the labels based on the embeddings


def no_embedding_protonet_method(study):
    abundance_data = pd.read_csv('/tudelft.net/staff-umbrella/abeellabstudents/sramezani/thesis/data/sun_et_al_data/mpa4_species_profile_preprocessed.csv', index_col=0, header=0)
    metadata = pd.read_csv("/tudelft.net/staff-umbrella/abeellabstudents/sramezani/thesis/data/sun_et_al_data/sample_group_species_preprocessed.csv", index_col=0, header=0)

    study_names: list = metadata["Project_1"].unique().tolist()
    test_metadata = metadata[metadata["Project_1"] == study]
    test_data = abundance_data.loc[test_metadata.index]
    train_metadata = metadata.drop(index=test_metadata.index)
    train_data = abundance_data.drop(index=test_data.index)

    resume_path = "/tudelft.net/staff-umbrella/abeellabstudents/sramezani/thesis/checkpoints/metalearning2/ProtoNet/{0}/TS{0}_TK10_unbalanced_sun et al_ProtoNet_Tspecies_"
    data_splits = "/tudelft.net/staff-umbrella/abeellabstudents/sramezani/thesis/data/sun_et_al_data/selected_data_splits/{0}/10shot_nOuter10_nInner5_{0}_69189786.yml"

    resume_path = resume_path.format(study)
    data_splits = data_splits.format(study)
    
    file_path = f"/tudelft.net/staff-umbrella/abeellabstudents/sramezani/thesis/res_analysis/{study}"
    # metalearning_res = get_desired_cross_val_results_for_models(
    #     "metalearning2",
    #     ["ProtoNet"],
    #     ["unbalanced", "sun et al", "10_shot"],
    #     "Outer fold.test/f1",
    #     ["LuW_2018"],  # Use all studies
    #     notes_contains="'extra_str_indicator': ''"
    #     )[0]
    # metalearning_res.columns = list(map(lambda x: "_".join(x.split("_")[:2])[2:], metalearning_res.columns))

    # study_scores = metalearning_res.loc[:, study].dropna()
    # study_scores = study_scores[study_scores != 0]
    # median = np.median(study_scores.values)
    # chosen_fold = min(study_scores.index, key=lambda k: abs(study_scores[k] - median))


    with open(data_splits, "r") as f:
        cross_val_data_selection = yaml.safe_load(f)
        test_loop_data_selection = cross_val_data_selection[0]

    checkpoint = load_checkpoint(str(resume_path + "/checkpoint.yaml"), True)

    storage_path = f"sqlite:///{resume_path}/optuna_study.db"
    storage = optuna.storages.RDBStorage(
        url=storage_path,
        heartbeat_interval=60,
        grace_period=120,
        failed_trial_callback=optuna.storages.RetryFailedTrialCallback(max_retry=1),
    )
    optuna_study = optuna.create_study(
        direction="maximize",
        study_name=f"hyper-param_optimization_for_{checkpoint['wandb_run_id']}",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
    )

    best_trial = optuna_study.best_trial
    best_trial_params = best_trial.params

    config = {
        "datasource": "sun et al",
        "algorithm": "ProtoNet",
        "balanced_or_unbalanced": "unbalanced",
        "resume": True,
        "positive_class_label": "Disease",
        "track_best_f1": True,
        "splitting_method": "study_wise",
        "feature_reduction_alg": "",
        "device": "cpu",
    }

    for fold, test_support_set in test_loop_data_selection.items():

        best_trial_config = protonet_search_space_sampler(best_trial)
        model, train_loader, val_loader, test_loader = get_metalearning_model_from_trial(
            train_data,
            test_data,
            train_metadata,
            test_metadata,  
            10,
            10,
            test_support_set,
            "ProtoNet",
            best_trial_config,
            config,
        )

        wandb_name = f"TS{study}_TK10_sun et al_protonet_Tspecies2"
        wandb.init(
            project="protonet_no_embedding",
            name=wandb_name,
            config=config,
            notes=str(config),
            group="ProtoNet",
            # tags=wandb_base_tags,
            # id=checkpoint["wandb_run_id"] if resume else None,
            # resume="allow" if resume and checkpoint["wandb_run_id"] else None,
        )

        for i, (X, y) in enumerate(test_loader):
            X, y = X.to(model.device), y.to(model.device, dtype=torch.int64)
            X_support = X[: model.eval_k_shot * 2, :]
            y_support = y[: model.eval_k_shot * 2]
            X_query = X[model.eval_k_shot * 2 :, :]
            y_query = y[model.eval_k_shot * 2 :]

            prototypes, classes = model.protonet.calculate_prototypes(X_support, y_support)
            preds, labels, dist = model.protonet.classify_feats(prototypes, classes, X_query, y_query)
            loss = F.cross_entropy(preds, labels, weight=model.class_weights_loss_fn)

            # to wandb
            results = {}
            results["loss"] = loss.item()
            metrics = compute_metrics(preds.argmax(dim=1), labels)
            for key in metrics:
                results[key] = float(metrics[key])


            eval_log = {
                    f"original/loss": results["loss"],
                    f"original/accuracy": results["accuracy"],
                    f"original/f1": results["f1"],
                    f"original/precision": results["precision"],
                    f"original/recall": results["recall"],
                    f"original/roc_auc": results["roc_auc"],
                    f"original/average_precision": results["average_precision"],
                    "fold": fold,
                }
            wandb.log(eval_log)



if __name__ == "__main__":
    fire.Fire()
