import numpy as np
import pandas as pd
import xgboost as xgb
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import Pool, CatBoostRegressor, CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, StratifiedKFold
import copy
import random
import math

from libs.models.supervised import filter_params


class simple_baseline:
    def __init__(self, modelname, tasktype, params, seed, cat_features=[], y_std=None):

        self.modelname = modelname
        self.tasktype = tasktype
        self.cat_features = cat_features
        self.seed = seed
        self.params = params
        self.y_std = y_std

        # -------------------------
        # HPO settings
        # -------------------------
        self.hpo = params.get("hpo", False)
        self.hpo_trials = params.get("hpo_trials", 0)
        self.cv_folds = params.get("cv_folds", 5)
        self.hpo_config = params.get("hpo_config", {})

        # Base model init
        self._init_model(params)

    # -------------------------
    # Model Initialization
    # -------------------------
    def _init_model(self, params):

        if self.modelname == "xgboost":
            pred_fn = {
                "multiclass": xgb.XGBClassifier,
                "binclass": xgb.XGBClassifier,
                "regression": xgb.XGBRegressor
            }

            loss_fn = {
                "multiclass": "multi:softmax",
                "binclass": "binary:logistic",
                "regression": "reg:squarederror"
            }

            params = filter_params(pred_fn[self.tasktype], params)

            self.model = pred_fn[self.tasktype](
                tree_method="hist",
                device="cuda",
                objective=loss_fn[self.tasktype],
                booster="gbtree",
                random_state=self.seed,
                solver="saga",
                **params
            )

        elif self.modelname == "catboost":
            pred_fn = {
                "multiclass": CatBoostClassifier,
                "binclass": CatBoostClassifier,
                "regression": CatBoostRegressor
            }

            loss_fn = {
                "multiclass": "MultiClass",
                "binclass": "CrossEntropy",
                "regression": "RMSE"
            }

            if params.get("data_id") == 1492:
                params["task_type"] = "GPU"
                params["devices"] = params.get("gpu_id", "0")

            params = filter_params(pred_fn[self.tasktype], params)

            self.model = pred_fn[self.tasktype](
                loss_function=loss_fn[self.tasktype],
                cat_features=self.cat_features,
                random_state=self.seed,
                verbose=0,
                **params
            )

        elif self.modelname == "lightgbm":
            loss_fn = {
                "multiclass": "multiclass",
                "binclass": "binary",
                "regression": "regression"
            }

            model_fn = {
                "multiclass": LGBMClassifier,
                "binclass": LGBMClassifier,
                "regression": LGBMRegressor
            }

            params = filter_params(model_fn[self.tasktype], params)

            self.model = model_fn[self.tasktype](
                objective=loss_fn[self.tasktype],
                verbose=-1,
                random_state=self.seed,
                **params
            )

        elif self.modelname == "lr":
            self.model = LogisticRegression(
                random_state=self.seed,
                penalty=params["penalty"],
                C=params["C"],
                max_iter=params["max_iter"],
                fit_intercept=params["fit_intercept"],
                solver="saga"
            )

        elif self.modelname == "knn":
            self.model = KNeighborsClassifier(n_neighbors=params["k"])

    # -------------------------
    # HPO Sampler
    # -------------------------
    def _sample_params(self):
        sampled = {}

        for key, cfg in self.hpo_config.items():
            if cfg["type"] == "int":
                sampled[key] = random.randint(cfg["low"], cfg["high"])

            elif cfg["type"] == "float":
                low, high = cfg["low"], cfg["high"]

                if cfg.get("log", False):
                    sampled[key] = math.exp(random.uniform(math.log(low), math.log(high)))
                else:
                    sampled[key] = random.uniform(low, high)

            elif cfg["type"] == "categorical":
                sampled[key] = random.choice(cfg["values"])

        return sampled

    # -------------------------
    # CV Evaluation
    # -------------------------
    def _evaluate(self, preds, y_true):
        if self.tasktype == "regression":
            return np.sqrt(np.mean((preds - y_true) ** 2))  # RMSE
        else:
            return np.mean(preds == y_true)

    def _run_cv(self, X, y):

        if self.tasktype in ["binclass", "multiclass"]:
            kf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
            splits = kf.split(X, y)
        else:
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
            splits = kf.split(X)

        scores = []

        for train_idx, val_idx in splits:

            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model_copy = copy.deepcopy(self.model)

            if self.modelname == "catboost":
                X_tr_df = pd.DataFrame(X_tr).astype({k: 'int' for k in self.cat_features})
                X_val_df = pd.DataFrame(X_val).astype({k: 'int' for k in self.cat_features})

                train_pool = Pool(X_tr_df, label=y_tr, cat_features=self.cat_features)
                model_copy.fit(train_pool)

                preds = model_copy.predict(X_val_df)

            elif self.modelname == "lightgbm":
                model_copy.fit(X_tr, y_tr, categorical_feature="auto")
                preds = model_copy.predict(X_val)

            else:
                model_copy.fit(X_tr, y_tr)
                preds = model_copy.predict(X_val)

            score = self._evaluate(preds, y_val)
            scores.append(score)

        return np.mean(scores)

    # -------------------------
    # HPO Loop
    # -------------------------
    def _run_hpo(self, X, y):

        best_score = float("inf")
        best_params = None

        base_params = self.params.copy()

        for trial in range(self.hpo_trials):

            sampled_params = self._sample_params()
            trial_params = {**base_params, **sampled_params}

            # rebuild model with trial params
            self._init_model(trial_params)

            score = self._run_cv(X, y)

            print(f"Trial {trial}: score={score:.5f}")

            if score < best_score:
                best_score = score
                best_params = trial_params

        print(f"Best CV score: {best_score:.5f}")
        return best_params

    # -------------------------
    # Fit
    # -------------------------
    def fit(self, X_train, y_train):

        X = X_train.cpu().numpy()
        y = y_train.cpu().numpy()

        labeled_flag = np.unique(np.where(~np.isnan(y))[0])
        X = X[labeled_flag]
        y = y[labeled_flag]

        if self.tasktype == "multiclass":
            y = np.argmax(y, axis=1)

        # -------------------------
        # HPO stage
        # -------------------------
        if self.hpo:
            best_params = self._run_hpo(X, y)
            self.params = best_params
            self._init_model(best_params)

        # -------------------------
        # Final training
        # -------------------------
        if self.modelname == "catboost":
            X_df = pd.DataFrame(X).astype({k: 'int' for k in self.cat_features})
            train_pool = Pool(X_df, label=y, cat_features=self.cat_features)
            return self.model.fit(train_pool)

        elif self.modelname == "lightgbm":
            return self.model.fit(X, y, categorical_feature="auto")

        else:
            return self.model.fit(X, y)

    # -------------------------
    # Predict
    # -------------------------
    def predict(self, X_test):
        X = X_test.cpu().numpy()

        if self.modelname in ["catboost", "lightgbm"]:
            X = pd.DataFrame(X).astype({k: 'int' for k in self.cat_features})

        return self.model.predict(X)

    def predict_proba(self, X_test, logit=False):
        X = X_test.cpu().numpy()

        if self.modelname in ["catboost", "lightgbm"]:
            X = pd.DataFrame(X).astype({k: 'int' for k in self.cat_features})

        if self.tasktype in ["binclass", "multiclass"]:
            return self.model.predict_proba(X)
        else:
            return None