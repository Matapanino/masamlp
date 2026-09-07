"""A weighted working-residual kernel head in the parent's logit units."""

import json

import numpy as np
from sklearn.metrics import roc_auc_score

from .kernels import _positive, _vector, _weights
from .nystrom_krr import NystromKRR


def _sigmoid(eta):
    eta = np.asarray(eta, dtype=np.float64)
    return np.exp(-np.logaddexp(0, -eta))


def _logit(p):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(p) - np.log1p(-p)


def _labels(y, n):
    y = _vector(y, n, "y")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y must contain binary labels 0 or 1")
    return y


class LinearResidualKRR:
    """Add a clipped global KRR working residual to supplied parent logits.

    There is no parent fit, preprocessing or internal validation split. The
    caller supplies every parent vector and selects gamma on training-internal
    validation rows. ``gamma_=0`` is the default, including immediately after fit.
    """

    def __init__(
        self, parent="logit", gamma_grid=(0, 0.125, 0.25, 0.5, 1.0), w_min=1e-4,
        clip_correction=4.0, **params,
    ):
        if parent not in ("logit", "proba"):
            raise ValueError("parent must be logit or proba")
        gamma_grid = tuple(float(gamma) for gamma in gamma_grid)
        if 0 not in gamma_grid or not np.isfinite(gamma_grid).all() or min(gamma_grid) < 0:
            raise ValueError("gamma_grid must contain 0 and only finite nonnegative values")
        _positive(w_min, "w_min")
        if np.isnan(clip_correction) or clip_correction <= 0:
            raise ValueError("clip_correction must be positive (infinity disables clipping)")
        self.parent = parent
        self.gamma_grid = tuple(sorted(set(gamma_grid)))
        self.w_min = float(w_min)
        self.clip_correction = float(clip_correction)
        self.params = NystromKRR(**params)._params()
        self.gamma_ = 0.0

    def fit(self, X, y, *, eta0=None, p0=None, sample_weight=None):
        """Fit z=(y-p0)/w with w=max(p0(1-p0), w_min)*sample_weight.

        Zero-weight rows receive z=0 and are omitted by the kernel solver.
        This is the literal working-target convention: sample weights enter
        both w and z's denominator, not just the loss weighting.
        """
        y = _labels(y, len(X))
        if self.parent == "logit":
            if eta0 is None or p0 is not None:
                raise ValueError("parent='logit' requires eta0 only")
            p = _sigmoid(_vector(eta0, len(X), "eta0"))
        else:
            if p0 is None or eta0 is not None:
                raise ValueError("parent='proba' requires p0 only")
            p = _vector(p0, len(X), "p0")
            if ((p < 0) | (p > 1)).any():
                raise ValueError("p0 must be in [0, 1]")
        sample_weight = _weights(sample_weight, len(X))
        w = np.maximum(p * (1 - p), self.w_min) * sample_weight
        z = np.divide(y - p, w, out=np.zeros_like(p), where=w > 0)
        solver = NystromKRR(**self.params).fit(X, z, w)
        self.solver_ = solver
        self.gamma_ = 0.0
        return self

    def correction(self, X):
        """Return the bounded logit correction, independent of gamma."""
        if not hasattr(self, "solver_"):
            raise RuntimeError("fit must be called before correction")
        return np.clip(self.solver_.predict(X), -self.clip_correction, self.clip_correction)

    def _gamma(self, gamma):
        gamma = self.gamma_ if gamma is None else gamma
        if not np.isscalar(gamma) or not np.isfinite(gamma) or gamma < 0:
            raise ValueError("gamma must be finite and nonnegative")
        return float(gamma)

    def predict_logit(self, X, eta0, gamma=None):
        """Return a parent copy at gamma=0, without evaluating the correction."""
        gamma = self._gamma(gamma)
        if gamma == 0:
            return np.array(eta0, copy=True)
        eta0 = np.asarray(eta0)
        if eta0.shape != (len(X),):
            raise ValueError("eta0 must be a vector aligned with X")
        with np.errstate(over="ignore", invalid="ignore"):
            return eta0 + gamma * self.correction(X)

    def predict_proba(self, X, p0=None, *, eta0=None, gamma=None):
        """Return positive-class probabilities, preserving supplied p0 at zero.

        Supply exactly one of p0 or eta0. Exact probability fallback requires
        p0 itself; converting logits through sigmoid cannot recover its bits.
        """
        gamma = self._gamma(gamma)
        if (p0 is None) == (eta0 is None):
            raise ValueError("supply exactly one of p0 or eta0")
        if p0 is not None:
            if gamma == 0:
                return np.array(p0, copy=True)
            p0 = _vector(p0, len(X), "p0")
            if ((p0 < 0) | (p0 > 1)).any():
                raise ValueError("p0 must be in [0, 1]")
            eta0 = _logit(p0)
        return _sigmoid(self.predict_logit(X, eta0, gamma))

    def select_gamma(self, X_val, eta0_val, y_val):
        """Maximize supplied-validation AUC; ties choose smaller gamma.

        Invalid predictions/metrics are discarded. An invalid baseline or
        single-class validation set conservatively retains the zero fallback.
        No training labels or validation arrays are retained by this method.
        """
        y = _labels(y_val, len(X_val))
        eta = np.asarray(eta0_val)
        if eta.shape != (len(X_val),):
            raise ValueError("eta0_val must be a vector aligned with X_val")
        self.gamma_ = 0.0
        self.gamma_scores_ = {}
        if not np.isfinite(eta).all() or len(np.unique(y)) != 2:
            return self.gamma_
        best = float(roc_auc_score(y, eta))
        self.gamma_scores_[0.0] = best
        for gamma in self.gamma_grid:
            if gamma == 0:
                continue
            prediction = self.predict_logit(X_val, eta, gamma)
            if not np.isfinite(prediction).all():
                self.gamma_scores_[gamma] = None
                continue
            score = float(roc_auc_score(y, prediction))
            self.gamma_scores_[gamma] = score if np.isfinite(score) else None
            if np.isfinite(score) and score > best:
                best, self.gamma_ = score, gamma
        return self.gamma_

    def save(self, path):
        """Save the solver, parent convention and selected gamma using np.savez."""
        if not hasattr(self, "solver_"):
            raise RuntimeError("fit must be called before save")
        metadata = {"parent": self.parent, "gamma_grid": self.gamma_grid, "w_min": self.w_min,
                    "clip_correction": self.clip_correction, "gamma": self.gamma_}
        with open(path, "wb") as file:
            np.savez(file, **self.solver_._state(), head=np.array(json.dumps(metadata)))

    @classmethod
    def load(cls, path, *, device=None):
        """Load the head, optionally overriding its cpu/cuda device."""
        with np.load(path, allow_pickle=False) as state:
            solver = NystromKRR._from_state(state, device=device)
            metadata = json.loads(str(state["head"]))
        gamma = metadata.pop("gamma")
        head = cls(**metadata, **solver._params())
        head.solver_ = solver
        head.gamma_ = head._gamma(gamma)
        return head
