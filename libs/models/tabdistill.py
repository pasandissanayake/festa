from torch.nn import functional as F
import einops

from tabpfn_extensions import TabPFNClassifier
from tabpfn_extensions.embedding import TabPFNEmbedding

import numpy as np
import time

import torch
import torch.nn as nn
from sklearn.preprocessing import label_binarize

from tqdm import tqdm
import numpy as np
import random
import inspect
from sklearn.model_selection import KFold


def check_softmax(logits):
    """
    Check if the logits are already probabilities, and if not, convert them to probabilities.
    
    :param logits: np.ndarray of shape (N, C) with logits
    :return: np.ndarray of shape (N, C) with probabilities
    """
    # Check if any values are outside the [0, 1] range and Ensure they sum to 1
    if np.any((logits < 0) | (logits > 1)) or (not np.allclose(logits.sum(axis=-1), 1, atol=1e-5)):
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # stabilize by subtracting max
        return exps / np.sum(exps, axis=1, keepdims=True)
    else:
        return logits


def dict_to_mlp(weight_dict: dict[str, torch.Tensor], in_dim:int) -> nn.Sequential:
    """
    Convert a dictionary of 'wbX' -> tensor(out_features, in_features+1)
    into a PyTorch MLP with the given weights and biases.
    
    Args:
        weight_dict: dict with keys like 'wb0', 'wb1', ... and tensors 
                     where the last column is the bias.
    
    Returns:
        model: nn.Sequential containing the layers with weights loaded.
    """
    layers = []
    
    # Sort layers by number (wb0, wb1, ...)
    sorted_keys = sorted(weight_dict.keys(), key=lambda k: int(k[2:]))
    
    for i, key in enumerate(sorted_keys):
        in_dim = in_dim + 1 # inputs and bias
        wb = weight_dict[key][0]
        out_dim = len(wb) // in_dim
        wb = torch.reshape(wb, (in_dim, out_dim))

        in_features = in_dim - 1  # last col = bias
        out_features = out_dim
        bias = wb[-1, :]
        weight = wb[:-1, :]
        
        # Create linear layer
        layer = nn.Linear(in_features, out_features)
        
        # Assign weights and bias (ensure no grad issues)
        with torch.no_grad():
            layer.weight.copy_(torch.transpose(weight, 0 ,1))
            layer.bias.copy_(bias)
        
        layers.append(layer)
        
        # Optionally add non-linearity (ReLU here, skip after last)
        if i < len(sorted_keys) - 1:
            layers.append(nn.ReLU())

        in_dim = out_dim
    
    return nn.Sequential(*layers)


class TabDistill():
    def __init__(self, params, tasktype, input_dim=0, output_dim=0, device='cuda', data_id=None, modelname="tabdistill"):
        # List of parameters
        # mlp_width -- width of the internal layers of the resultant MLP
        # mlp_depth -- number of internal layers
        # batch_size -- training batch size
        # learning_rate -- learning rate for training the regressor
        # weight_decay -- weight decay for traning the regressor
        # lr_scheduler -- boolean, use scheduler if true
        self.params = params
        self.tasktype = tasktype
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.data_id = data_id
        self.modelname = modelname
        self.params = params

        self.model = TabDistillClassifier(
            params=self.params,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            device=self.device
        )

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
            print("Starting HPO...")
            return self._fit_with_hpo(X_train, y_train)
        else:
            print("Starting single fit()...")
            return self._fit_single(X_train, y_train)
        
    
    def _fit_single(self, X_train, y_train):
        if y_train.ndim == 2:
            X_train = X_train[~torch.isnan(y_train[:, 0])]
            y_train = y_train[~torch.isnan(y_train[:, 0])]
        else:
            X_train = X_train[~torch.isnan(y_train)]
            y_train = y_train[~torch.isnan(y_train)]
            
        if y_train.ndim == 1:
            y_train = y_train.unsqueeze(1)

        if self.tasktype == "multiclass":
            y_train = y_train.argmax(axis=1)


        self.model.to(self.device)
        self.model(X_train.to(self.device), y_train.to(self.device))
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.model.apply(init_weights)

        n_queries = len(X_train)

        print(f"TabPFN regressor input dimension: {self.model.regressor[1].in_features}, total hyponet parameters: {self.model.total_params}, # queries: {n_queries}")

        print(f"Device: {self.device}")


        # Prepare data
        batch_size = self.params.get("batch_size", 100)

        if self.tasktype == "regression":
            loss_fn = torch.nn.functional.mse_loss
        elif self.tasktype == "binclass":
            loss_fn = torch.nn.functional.binary_cross_entropy_with_logits
        else:
            loss_fn = torch.nn.functional.cross_entropy

        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        
        if len(dataset) % batch_size == 1:
            loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size, shuffle=True, drop_last=True)
        else:
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Prepare optimizer
        optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.params["learning_rate"], weight_decay=self.params["weight_decay"])
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

                x = x.to(self.device)
                y = y.to(self.device)

                hyponet = self.model(x, y)
                out = hyponet(x)

                loss = loss_fn(out, y.to(self.device))
                loss.backward()

                optimizer.step()

                if self.params.get("lr_scheduler", False):
                    scheduler.step()

                pbar.set_postfix_str(
                    f"data_id: {self.data_id}, Model: {self.modelname}, Tr loss: {loss:.5f}"
                )

        self.model.eval()
        self.hyponet = dict_to_mlp(self.model(X_train, y_train).params, self.input_dim)
        return self


    def _fit_with_hpo(self, X_train, y_train):

        hpo_config = self.params["hpo_config"]
        n_trials = self.params.get("hpo_trials", 20)
        n_splits = self.params.get("cv_folds", 4)

        print(f"Starting HPO with {n_trials} trials, {n_splits}-fold CV")
        print(f"HPO config: {hpo_config}")

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
        model = TabDistillClassifier(
            params=trial_params,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            device=self.device
        )
        return model

    # =========================
    # Prediction
    # =========================
    def predict(self, X_test, cat_features=[]):
        X_test = X_test.to(self.device)
        self.hyponet.to(self.device)
        with torch.no_grad():
            if X_test.shape[0] > 10000:
                logits = []
                iters = X_test.shape[0] // 100 + 1
                for i in range(iters):
                    pred = self.hyponet(X_test[100*i:100*(i+1)])
                    logits.append(pred)
                logits = torch.concatenate(logits, dim=0)
            else:
                logits = self.hyponet(X_test)

            if self.tasktype == "binclass":
                return torch.sigmoid(logits).round().cpu().numpy()
            elif self.tasktype == "regression":
                return logits.cpu().numpy()
            else:
                return torch.argmax(logits, dim=1).cpu().numpy()

    def predict_proba(self, X_test, cat_features=[], logit=False):
        X_test = X_test.to(self.device)
        self.hyponet.to(self.device)
        with torch.no_grad():
            if X_test.shape[0] > 10000 or X_test.shape[1] > 240:
                logits = []
                iters = X_test.shape[0] // 100 + 1
                for i in range(iters):
                    pred = self.hyponet(X_test[100*i:100*(i+1)])
                    logits.append(pred)
                logits = torch.concatenate(logits, dim=0)
            else:
                logits = self.hyponet(X_test)

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





class TabDistillClassifier(nn.Module):
    def __init__(self,
                 params,
                 input_dim,
                 output_dim,
                 device,
                ):
        super().__init__()        
        
        
        self.hyponet = None
        self.params = params
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

        self.classifier = TabPFNClassifier(device=device, n_estimators=1)
        self.extractor = TabPFNEmbedding(tabpfn_clf=self.classifier, n_fold=0)
        
        self.hyponet = HypoMlp(
            depth=self.params["mlp_depth"],
            in_dim=self.input_dim,
            out_dim=self.output_dim,
            hidden_dim=self.params["mlp_width"]
        )
        
        total_params = 0
        for name, shape in self.hyponet.param_shapes.items():
            total_params += shape[0] * shape[1]
        
        self.total_params = total_params

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(self.total_params),
            nn.LayerNorm(self.total_params),
        )
            
    def forward(self, queries_x, queries_y):
        X = queries_x.cpu()
        y = queries_y.cpu().float()

        self.classifier.fit(X, y)
        embeddings = self.extractor.get_embeddings(X, y, X, data_source="test")
        embeddings = einops.rearrange(embeddings, "batch sample features -> batch (sample features)")
        outputs = torch.tensor(embeddings).cuda()
        outputs = self.regressor(outputs)
                
        params = dict()
        start_idx = 0
        for name, shape in self.hyponet.param_shapes.items():
            end_idx = start_idx + shape[0] * shape[1]
            wb = F.normalize(outputs[:, start_idx:end_idx], dim=1)
            params[name] = wb
            start_idx = end_idx
        self.hyponet.set_params(params=params)
        return self.hyponet
    

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


class Scaler(nn.Module):
    def __init__(self, init_scale):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))

    def forward(self, x):
        return x * self.scale


class HypoMlp(nn.Module):
    def __init__(self,
                 depth,
                 in_dim,
                 out_dim,
                 hidden_dim):
        super().__init__()
        self.depth = depth
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        
        # create parameter shapes dict()
        self.param_shapes = dict()
        for i in range(self.depth):
            d1 = self.hidden_dim + 1 if i > 0 else self.in_dim + 1
            d2 = self.hidden_dim if i < self.depth - 1 else self.out_dim
            self.param_shapes[f'wb{i}'] = (d1, d2)

        self.relu = nn.ReLU()
        self.params = {}

    def set_params(self, params):
        self.params = params

    def forward(self, x):
        x = x.unsqueeze(dim=0)
        for i in range(self.depth):
            x = batched_linear_mm(x, self.params[f'wb{i}'])
            if i < self.depth - 1:
                x = self.relu(x)
        x = x.squeeze(dim=0)
        return x


def batched_linear_mm(x, wb):
    # args shapes --> x: (batch, n_queries, D1); wb: (batch, (D1 + 1) x D2)
    one = torch.ones(*x.shape[:-1], 1, device=x.device)
    x, _ = einops.pack([x, one], "batch n_queries *")
    wb = einops.rearrange(wb, "batch (in_dim out_dim) -> batch in_dim out_dim", in_dim=x.shape[2])
    wb = einops.repeat(wb, "batch in_dim out_dim -> batch n_queries in_dim out_dim", n_queries=x.shape[1])
    return einops.einsum(x, wb, "batch n_queries in_dim, batch n_queries in_dim out_dim -> batch n_queries out_dim")