import torch
from tqdm import tqdm
import numpy as np
import random
import inspect
from sklearn.model_selection import KFold


# =========================
# Utilities
# =========================

def filter_params(func, params):
    signature = inspect.signature(func)
    valid_params = signature.parameters.keys()
    return {k: v for k, v in params.items() if k in valid_params}


def sample_hyperparameters(hpo_config):
    sampled = {}
    for key, spec in hpo_config.items():
        if spec["type"] == "loguniform":
            sampled[key] = float(
                np.exp(np.random.uniform(np.log(spec["low"]), np.log(spec["high"])))
            )
        elif spec["type"] == "uniform":
            sampled[key] = float(np.random.uniform(spec["low"], spec["high"]))
        elif spec["type"] == "int":
            sampled[key] = int(np.random.randint(spec["low"], spec["high"] + 1))
        elif spec["type"] == "categorical":
            sampled[key] = random.choice(spec["values"])
    return sampled


def train_val_split(X, y, val_ratio=0.2):
    n = X.shape[0]
    idx = torch.randperm(n)
    val_size = int(n * val_ratio)

    val_idx = idx[:val_size]
    train_idx = idx[val_size:]

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


# =========================
# Learning Rate Scheduler
# =========================

class CosineAnnealingLR_Warmup(object):
    def __init__(self, optimizer, warmup_epochs, T_max, iter_per_epoch,
                 base_lr, warmup_lr, eta_min, last_epoch=-1):

        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.T_max = T_max
        self.iter_per_epoch = iter_per_epoch
        self.base_lr = base_lr
        self.warmup_lr = warmup_lr
        self.eta_min = eta_min
        self.last_epoch = last_epoch

        self.warmup_iter = self.iter_per_epoch * self.warmup_epochs
        self.cosine_iter = self.iter_per_epoch * (self.T_max - self.warmup_epochs)
        self.current_iter = (self.last_epoch + 1) * self.iter_per_epoch

        self.step()

    def get_current_lr(self):
        if self.current_iter < self.warmup_iter:
            current_lr = (self.base_lr - self.warmup_lr) / self.warmup_iter * self.current_iter + self.warmup_lr
        else:
            current_lr = self.eta_min + (
                self.base_lr - self.eta_min
            ) * (1 + np.cos(np.pi * (self.current_iter - self.warmup_iter) / self.cosine_iter)) / 2
        return current_lr

    def step(self):
        current_lr = self.get_current_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr
        self.current_iter += 1


def CosineAnnealingParam(warmup_epochs, T_max, iter_per_epoch,
                         current_iter, base_value,
                         warmup_value=1e-8, eta_min=0):

    warmup_iter = iter_per_epoch * warmup_epochs
    cosine_iter = iter_per_epoch * (T_max - warmup_epochs)

    if current_iter < warmup_iter:
        return (base_value - warmup_value) / warmup_iter * current_iter + warmup_value
    else:
        return eta_min + (base_value - eta_min) * (
            1 + np.cos(np.pi * (current_iter - warmup_iter) / cosine_iter)
        ) / 2


# =========================
# Supervised Base Model
# =========================

class supmodel(torch.nn.Module):
    def __init__(self, params, tasktype, device,
                 data_id=None, modelname=None,
                 cat_features=[],
                 model_class=None,
                 model_init_params=None):

        super(supmodel, self).__init__()

        self.tasktype = tasktype
        self.cat_features = cat_features
        self.device = device
        self.params = params
        self.data_id = data_id
        self.modelname = modelname

        self.model_class = model_class
        self.model_init_params = model_init_params or {}

    # =========================
    # Public Fit Interface
    # =========================
    def fit(self, X_train, y_train):
        # Handle NaNs
        if y_train.ndim == 2:
            mask = ~torch.isnan(y_train[:, 0])
        else:
            mask = ~torch.isnan(y_train)

        X_train = X_train[mask]
        y_train = y_train[mask]

        print(f"Few-shot train shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")

        if self.params.get("hpo", False):
            return self._fit_with_hpo(X_train, y_train)
        else:
            return self._fit_single(X_train, y_train)

    # =========================
    # Single Training Run
    # =========================
    def _fit_single(self, X_train, y_train):

        print(f"Device: {self.device}")

        batch_size = self.params.get("batch_size", 100)

        optimizer = self.model.make_optimizer()

        if self.tasktype == "regression":
            loss_fn = torch.nn.functional.mse_loss
        elif self.tasktype == "binclass":
            loss_fn = torch.nn.functional.binary_cross_entropy_with_logits
        else:
            loss_fn = torch.nn.functional.cross_entropy

        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        del X_train, y_train

        if len(dataset) % batch_size == 1:
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                 shuffle=True, drop_last=True)
        else:
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                 shuffle=True)

        optimizer.zero_grad()
        optimizer.step()

        if self.params.get("lr_scheduler", False):
            scheduler = CosineAnnealingLR_Warmup(
                optimizer,
                base_lr=self.params['learning_rate'],
                warmup_epochs=10,
                T_max=self.params.get('n_epochs'),
                iter_per_epoch=len(loader),
                warmup_lr=1e-6,
                eta_min=0,
                last_epoch=-1
            )

        self.model.to(self.device)

        pbar = tqdm(range(1, self.params.get('n_epochs', 0) + 1))

        for epoch in pbar:
            pbar.set_description(f"EPOCH: {epoch}")

            for x, y in loader:
                self.model.train()
                optimizer.zero_grad()

                out = self.model(x.to(self.device), self.cat_features)

                if out.size() != y.size():
                    out = out.view(y.size())

                loss = loss_fn(out, y.to(self.device))
                loss.backward()

                optimizer.step()

                if self.params.get("lr_scheduler", False):
                    scheduler.step()

                pbar.set_postfix_str(
                    f"data_id: {self.data_id}, Model: {self.modelname}, Tr loss: {loss:.5f}"
                )

        self.model.eval()

    # =========================
    # HPO Wrapper
    # =========================
    def _fit_with_hpo(self, X_train, y_train):

        hpo_config = self.params["hpo_config"]
        n_trials = self.params.get("hpo_trials", 20)
        n_splits = self.params.get("cv_folds", 4)

        print(f"Starting HPO with {n_trials} trials, {n_splits}-fold CV")
        print(f"HPO config: {hpo_config}")

        unique, counts = np.unique(y_train.cpu().numpy(), return_counts=True)
        min_class_count = counts.min()

        # adjust number of splits
        n_splits = min(n_splits, min_class_count)

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.params.get("seed", 0))

        best_score = -float("inf")
        best_params = None

        for trial in range(n_trials):

            trial_params = sample_hyperparameters(hpo_config)

            fold_scores = []

            print(f"\n--- Trial {trial} ---")
            print(f"Params: {trial_params}")

            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):

                # Split data
                X_tr = X_train[train_idx]
                y_tr = y_train[train_idx]
                X_val = X_train[val_idx]
                y_val = y_train[val_idx]

                # Backup original params
                original_params = self.params.copy()
                self.params.update(trial_params)

                # 🔥 IMPORTANT: rebuild model for each fold
                self.model = self._build_model_with_params(trial_params)

                # Train
                self._fit_single(X_tr, y_tr)

                # Evaluate
                preds = self.predict(X_val)
                score = self._evaluate(preds, y_val)

                fold_scores.append(score)

                print(f"  Fold {fold_idx}: score={score:.5f}")

                # Restore params
                self.params = original_params

            # Aggregate across folds
            mean_score = float(np.mean(fold_scores))
            std_score = float(np.std(fold_scores))

            print(f"Trial {trial}: mean={mean_score:.5f}, std={std_score:.5f}")

            # Optional: variance-aware selection
            final_score = mean_score - 0.1 * std_score

            if final_score > best_score:
                best_score = final_score
                best_params = trial_params

        print("\nBest params:", best_params)

        # 🔥 Retrain on full dataset
        self.params.update(best_params)
        self.model = self._build_model_with_params(best_params)
        self._fit_single(X_train, y_train)

        return self

    # =========================
    # Model Builder Hook
    # =========================
    def _build_model_with_params(self, trial_params):
        # Merge base params + trial params
        full_params = {**self.model_init_params, **trial_params}

        # Only pass valid args to constructor
        valid_params = filter_params(self.model_class.__init__, full_params)

        model = self.model_class(**valid_params)
        return model

    # =========================
    # Prediction
    # =========================
    def predict(self, X_test, cat_features=[]):
        with torch.no_grad():
            if X_test.shape[0] > 10000:
                logits = []
                iters = X_test.shape[0] // 100 + 1
                for i in range(iters):
                    pred = self.model(X_test[100*i:100*(i+1)], cat_features)
                    logits.append(pred)
                logits = torch.concatenate(logits, dim=0)
            else:
                logits = self.model(X_test, cat_features)

            if self.tasktype == "binclass":
                return torch.sigmoid(logits).round().cpu().numpy()
            elif self.tasktype == "regression":
                return logits.cpu().numpy()
            else:
                return torch.argmax(logits, dim=1).cpu().numpy()

    def predict_proba(self, X_test, cat_features=[], logit=False):
        with torch.no_grad():
            if X_test.shape[0] > 10000 or X_test.shape[1] > 240:
                logits = []
                iters = X_test.shape[0] // 100 + 1
                for i in range(iters):
                    pred = self.model(X_test[100*i:100*(i+1)], cat_features)
                    logits.append(pred)
                logits = torch.concatenate(logits, dim=0)
            else:
                logits = self.model(X_test, cat_features)

            if logit:
                return logits
            elif self.tasktype == "binclass":
                return torch.sigmoid(logits).cpu().numpy()
            elif self.tasktype == "multiclass":
                return torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
            else:
                return logits.cpu().numpy()

    # =========================
    # Evaluation Helper
    # =========================
    def _evaluate(self, preds, y_val):
        y_true = y_val.cpu().numpy()
        if self.tasktype == "regression":
            return -np.mean((preds - y_true) ** 2)
        elif self.tasktype == "binclass":
            return (preds == y_true).mean()
        else:
            y_true = np.argmax(y_true, axis=1)
            return (preds == y_true).mean()