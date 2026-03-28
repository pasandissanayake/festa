import optuna
import yaml
import numpy as np
from copy import deepcopy
import tempfile
import os
from joblib import Parallel, delayed
import datetime, time, argparse

# Import FESTA main
from main import main as festa_main


# -----------------------------
# SETTINGS
# -----------------------------
SEEDS = [0, 1, 2, 3, 4]
DATASET_ID = 31
SHOT = 5
GPU_ID = 0
N_JOBS = len(SEEDS)   # parallel seeds


# -----------------------------
# Load YAML
# -----------------------------
def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# -----------------------------
# Set nested config values
# -----------------------------
def set_nested(config, key, value):
    keys = key.split(".")
    d = config
    for k in keys[:-1]:
        if k not in d:
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value


# -----------------------------
# Sample params from config
# -----------------------------
def sample_from_config(trial, search_space):
    params = {}

    for name, spec in search_space.items():

        if spec["type"] == "float":
            params[name] = trial.suggest_float(
                name,
                spec["low"],
                spec["high"],
                log=spec.get("log", False)
            )

        elif spec["type"] == "int":
            params[name] = trial.suggest_int(
                name,
                spec["low"],
                spec["high"]
            )

        elif spec["type"] == "categorical":
            params[name] = trial.suggest_categorical(
                name,
                spec["choices"]
            )

        else:
            raise ValueError(f"Unknown param type: {spec['type']}")

    return params


# -----------------------------
# Metric extraction
# -----------------------------
def extract_metric(test_score, tasktype):

    # SSL models return nested dict
    if isinstance(test_score, dict) and "finetuning" in test_score:
        test_score = test_score["finetuning"]

    if tasktype == "binclass":
        return test_score["auc"], "maximize"

    elif tasktype == "multiclass":
        return test_score["accuracy"], "maximize"

    elif tasktype == "regression":
        return test_score["rmse"], "minimize"

    else:
        raise ValueError(f"Unknown tasktype: {tasktype}")


# -----------------------------
# Get tasktype once
# -----------------------------
def get_tasktype():
    import json
    with open("dataset_id.json", "r") as f:
        data_info = json.load(f)
    return data_info[str(DATASET_ID)]["tasktype"]


TASKTYPE = get_tasktype()


# -----------------------------
# Run one seed
# -----------------------------
def run_single_seed(config_path, seed, trial_id):

    festa_args = argparse.Namespace(
        gpu_id=GPU_ID,
        openml_id=DATASET_ID,
        shot=SHOT,
        seed=seed,
        force_train=True,
        config_filename=config_path,
        trial_id=trial_id   # 🔥 important
    )

    test_score = festa_main(festa_args)

    if test_score is None:
        raise RuntimeError(f"Run failed for seed {seed}")

    metric, _ = extract_metric(test_score, TASKTYPE)

    return metric


# -----------------------------
# Objective
# -----------------------------
def objective(trial):

    base_config = load_yaml("configs/mlp.yaml")
    hpo_config = load_yaml("configs_hpo/mlp.yaml")

    sampled = sample_from_config(trial, hpo_config["search_space"])

    # Apply params
    config = deepcopy(base_config)
    for k, v in sampled.items():
        set_nested(config, k, v)

    # Save temp config
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml") as f:
        yaml.dump(config, f)
        config_path = f.name

    # -----------------------------
    # Parallel multi-seed execution
    # -----------------------------
    try:
        scores = Parallel(n_jobs=N_JOBS)(
            delayed(run_single_seed)(config_path, seed, trial.number)
            for seed in SEEDS
        )
    except Exception as e:
        os.remove(config_path)
        raise e

    os.remove(config_path)

    mean = float(np.mean(scores))
    std = float(np.std(scores))

    trial.set_user_attr("scores", scores)
    trial.set_user_attr("mean", mean)
    trial.set_user_attr("std", std)

    print(f"[Trial {trial.number}] mean={mean:.4f}, std={std:.4f}")

    print(scores)

    # -----------------------------
    # Variance-aware objective
    # -----------------------------

    return mean - 0.1 * std


# -----------------------------
# Run study
# -----------------------------
if __name__ == "__main__":

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler()
    )

    study.optimize(objective, n_trials=2)

    print("\nBest params:")
    print(study.best_params)

    print("\nBest value:")
    print(study.best_value)

    with open("best_params.yaml", "w") as f:
        yaml.dump(study.best_params, f)