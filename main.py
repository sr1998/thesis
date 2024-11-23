import os
from pathlib import Path
from typing import Dict, Union
from sklearn.pipeline import Pipeline
from joblib import Memory
import yaml
from loguru import logger 

from mgnify_helper_functions import MGnifyData


    



def build_preprocessor_pipeline(config: Dict) -> Pipeline:
    # Create a pipeline for batch correction methods
    p = []
    for batch_corrector in config["batch_correction_methods"]:
        pass
    batch_correction_methods = Pipeline(p)

    # Create a pipeline for imputation methods
    p = []
    for imputation in config["imputations"]:
        pass
    imputations = Pipeline(p)

    # Create a pipeline for normalization and transformation methods 
    p = []
    for n_or_t in config["normalizations_and_transformations"]:
        if n_or_t == "total-sum scaling":
            p.append((n_or_t, ))
    normalizations_and_transformations = Pipeline(p)

    # Create a pipeline for feature space changes
    p = []
    for feature_space_change in config["feature_space_changes"]:
        pass
    feature_space_changes = Pipeline(p)

    return Pipeline([
        ("batch_correction_methods", batch_correction_methods),
        ("imputations", imputations),
        ("normalizations_and_transformations", normalizations_and_transformations),
        ("feature_space_changes", feature_space_changes)
    ])


def main():
    config_path = "config.yaml"
    config_file_name = os.path.basename(config_path).split(".")[0]
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # get misc config parameters
    data_dir = config["data_dir"]
    wandb = config["wandb"]
    verbose_pipeline = config["verbose_pipeline"]
    cache_pipeline_steps = config["cache_pipeline_steps"]

    # Set up file logging
    logger_path = os.path.join(data_dir, "logs", f"pipeline_{config_file_name}.log")
    logger.add(logger_path, colorize=True, level="INFO")

    # Initialize wandb if enabled
    if wandb:
        wandb.init(project="thesis", name=config_file_name)
    else:
        wandb.init(mode="disabled", project="thesis", name=config_file_name)
    
    # Create a cache object if caching is enabled with default cache location
    if cache_pipeline_steps:
        cacher = Memory(location=os.path.join(data_dir, "cache"), verbose=1)
    else:
        cacher = None

    apply_data_pipeline(config["data"], data_dir)
    preprocessor_pipeline = build_preprocessor_pipeline(config["preprocessing"], cacher)
    model_pipeline = build_model_pipeline(config["model"], cacher)
    
    # Load the data

if __name__ == "__main__":