import time

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin


class TorchLinearSVC(ClassifierMixin, BaseEstimator):
    def __init__(self, C=1.0, standardize=True, max_iter=300, tol=1e-6,
                 device="auto", chunk_size=65536, verbose=True):
        self.C = C
        self.standardize = standardize
        self.max_iter = max_iter
        self.tol = tol
        self.device = device
        self.chunk_size = chunk_size
        self.verbose = verbose

    def _dev(self):
        from src.gpu_features import resolve_device
        d = resolve_device(self.device)
        return d

    @staticmethod
    def _fits_in_gpu(nbytes, dev):
        if dev.type != "cuda":
            return True
        free, _ = torch.cuda.mem_get_info(dev)
        return nbytes < 0.5 * free

    def fit(self, X, y, sample_weight=None):
        dev = self._dev()
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n, d = X.shape
        k = len(self.classes_)
        t0 = time.time()

        Xt = torch.from_numpy(X)
        on_gpu = self._fits_in_gpu(Xt.numel() * 4, dev)
        Xt = Xt.to(dev) if on_gpu else Xt.pin_memory() if dev.type == "cuda" else Xt
        chunks = [(i, min(i + self.chunk_size, n)) for i in range(0, n, self.chunk_size)]

        def get(a, b):
            return Xt[a:b].to(dev, non_blocking=True)

        if self.standardize:
            s1 = torch.zeros(d, dtype=torch.float64, device=dev)
            s2 = torch.zeros(d, dtype=torch.float64, device=dev)
            for a, b in chunks:
                xc = get(a, b).double()
                s1 += xc.sum(0)
                s2 += (xc * xc).sum(0)
            mean = s1 / n
            std = torch.sqrt((s2 / n - mean ** 2).clamp_min(0))
            std[std < 1e-12] = 1.0
            mean, std = mean.float(), std.float()
        else:
            mean = torch.zeros(d, device=dev)
            std = torch.ones(d, device=dev)

        Y = -torch.ones(n, k, device=dev)
        Y[torch.arange(n), torch.from_numpy(y_idx).to(dev)] = 1.0

        if sample_weight is None:
            sw = torch.ones(n, device=dev)
        else:
            sw = torch.as_tensor(np.asarray(sample_weight, dtype=np.float32), device=dev)
            sw = sw * (n / sw.sum())

        W = torch.zeros(d, k, device=dev, requires_grad=True)
        b = torch.zeros(k, device=dev, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], lr=1, max_iter=self.max_iter, tolerance_grad=self.tol,
                                tolerance_change=1e-10, history_size=20,
                                line_search_fn="strong_wolfe")
        reg = 0.5 / (self.C * n)
        n_eval = [0]

        def closure():
            opt.zero_grad()
            total = 0.0
            for a, bb in chunks:
                xc = (get(a, bb) - mean) / std
                margin = 1 - Y[a:bb] * (xc @ W + b)
                loss = ((torch.clamp(margin, min=0) ** 2).sum(1) * sw[a:bb]).sum() / n
                loss.backward()
                total += loss.item()
            r = reg * (W * W).sum()
            r.backward()
            n_eval[0] += 1
            return torch.tensor(total + r.item())

        opt.step(closure)
        if self.verbose:
            print(f"  SVM lineal en {dev.type.upper()}: {n_eval[0]} evaluaciones, "
                  f"{time.time() - t0:.1f} s")

        self.coef_ = W.detach().T.cpu().numpy()
        self.intercept_ = b.detach().cpu().numpy()
        self.mean_ = mean.cpu().numpy()
        self.scale_ = std.cpu().numpy()
        self.n_features_in_ = d
        del Xt, Y, sw
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float32)
        try:
            dev = self._dev()
        except RuntimeError:
            dev = torch.device("cpu")
        W = torch.from_numpy(self.coef_.T).to(dev)
        b = torch.from_numpy(self.intercept_).to(dev)
        mean = torch.from_numpy(self.mean_).to(dev)
        std = torch.from_numpy(self.scale_).to(dev)
        out = []
        with torch.no_grad():
            for i in range(0, len(X), self.chunk_size):
                xc = (torch.from_numpy(X[i:i + self.chunk_size]).to(dev) - mean) / std
                out.append((xc @ W + b).cpu().numpy())
        s = np.concatenate(out) if out else np.zeros((0, len(self.classes_)), np.float32)
        return s[:, 1] - s[:, 0] if len(self.classes_) == 2 else s

    def predict(self, X):
        s = self.decision_function(X)
        idx = (s > 0).astype(int) if s.ndim == 1 else s.argmax(1)
        return self.classes_[idx]


def build_model(name, seed, max_depth=None, device="auto", svm_c=1.0, svm_iters=300,
                svm_impl="torch"):
    from src.gpu_features import resolve_device
    dev = resolve_device(device)
    if name == "svm":
        if svm_impl == "sklearn":
            from sklearn.svm import SVC
            return SVC(kernel="linear")
        return TorchLinearSVC(C=svm_c, max_iter=svm_iters, device=dev.type)
    if name == "tree":
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                         eval_metric="mlogloss", random_state=seed,
                         tree_method="hist", device=dev.type)


def prepare_for_inference(model, device="auto"):
    from src.gpu_features import resolve_device
    try:
        dev = resolve_device(device)
    except RuntimeError:
        dev = torch.device("cpu")
    if hasattr(model, "get_xgb_params"):
        model.set_params(device=dev.type)
    elif isinstance(model, TorchLinearSVC):
        model.device = dev.type
    return model
