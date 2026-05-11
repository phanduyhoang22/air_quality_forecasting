"""
src/models/ridge_regression.py
================================
Ridge Regression — Implemented from Scratch via Mini-Batch Gradient Descent

Mathematical Formulation
------------------------

Objective (L2-Regularised MSE):
    L(w, b) = (1/2n) · ‖Xw + b·1 − y‖² + (λ/2) · ‖w‖²

where:
    X ∈ ℝ^{n×p}  — feature matrix
    w ∈ ℝ^p      — weight vector  (regularised)
    b ∈ ℝ        — bias / intercept (NOT regularised)
    y ∈ ℝ^n      — target vector
    λ ≥ 0        — regularisation strength

Gradient:
    ∂L/∂w = (1/n) · Xᵀ(Xw + b − y) + λ·w
    ∂L/∂b = (1/n) · Σᵢ (ŷᵢ − yᵢ)

Note on bias exclusion from penalty:
    The bias term b captures the global mean of y.  Including it in ‖w‖²
    would pull the intercept toward zero, introducing systematic bias
    (bias-in-the-statistical-sense) unrelated to regularisation.
    This matches the convention in sklearn.linear_model.Ridge(fit_intercept=True).

Update rule (mini-batch SGD):
    w ← w − η · ∂L/∂w
    b ← b − η · ∂L/∂b

Early Stopping:
    Track val_loss each epoch.  If val_loss has not improved by ≥ tol for
    `patience` consecutive epochs, halt and restore best weights.
    This acts as an implicit regulariser in the iteration-count dimension.

Vectorisation note:
    All matrix operations use NumPy broadcasting — no Python loops over
    samples.  Complexity per epoch: O(n·p) for the matrix-vector products.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class RidgeRegression:
    """
    Mini-batch Gradient Descent Ridge Regression.

    Parameters
    ----------
    alpha         : float — L2 regularisation coefficient λ  (default 1.0)
    learning_rate : float — step size η                      (default 0.01)
    max_epochs    : int   — maximum number of full passes     (default 2000)
    batch_size    : int   — mini-batch size                   (default 512)
    patience      : int   — early-stopping patience          (default 30)
    tol           : float — min relative improvement for ES  (default 1e-6)
    random_state  : int   — RNG seed for reproducibility     (default 42)
    """

    def __init__(
        self,
        alpha:         float = 1.0,
        learning_rate: float = 0.01,
        max_epochs:    int   = 2000,
        batch_size:    int   = 512,
        patience:      int   = 30,
        tol:           float = 1e-6,
        random_state:  int   = 42,
    ) -> None:
        self.alpha         = alpha
        self.learning_rate = learning_rate
        self.max_epochs    = max_epochs
        self.batch_size    = batch_size
        self.patience      = patience
        self.tol           = tol
        self.random_state  = random_state

        # Learned parameters
        self.w_: Optional[np.ndarray] = None   # shape (p,)
        self.b_: float = 0.0

        # Training history
        self.train_losses_: List[float] = []
        self.val_losses_:   List[float] = []
        self.best_epoch_:   int = 0

    # -----------------------------------------------------------------------
    # Core computations
    # -----------------------------------------------------------------------

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        """
        Compute ŷ = X·w + b.

        Parameters
        ----------
        X : np.ndarray of shape (n, p)

        Returns
        -------
        np.ndarray of shape (n,)
        """
        return X @ self.w_ + self.b_

    def _mse_loss(self, residuals: np.ndarray) -> float:
        """
        MSE = (1/n) · ‖r‖²   (without regularisation, used for monitoring).

        Parameters
        ----------
        residuals : ŷ − y  of shape (n,)

        Returns
        -------
        float scalar
        """
        return float(np.mean(residuals ** 2))

    def _ridge_loss(self, residuals: np.ndarray) -> float:
        """
        Full Ridge objective: (1/n)‖r‖² + λ‖w‖².

        Parameters
        ----------
        residuals : ŷ − y of shape (n,)

        Returns
        -------
        float scalar
        """
        return self._mse_loss(residuals) + self.alpha * float(np.dot(self.w_, self.w_))

    # -----------------------------------------------------------------------
    # Gradient computation (vectorised)
    # -----------------------------------------------------------------------

    def _gradients(
        self, X: np.ndarray, residuals: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Compute ∂L/∂w and ∂L/∂b for a mini-batch.

        ∂L/∂w = (2/n)·Xᵀr + 2λ·w
        ∂L/∂b = (2/n)·Σr

        Parameters
        ----------
        X         : mini-batch features   (n_batch, p)
        residuals : ŷ − y for mini-batch  (n_batch,)

        Returns
        -------
        (grad_w, grad_b) tuple
        """
        n = len(residuals)
        grad_w = (2.0 / n) * (X.T @ residuals) + 2.0 * self.alpha * self.w_
        grad_b = (2.0 / n) * residuals.sum()
        return grad_w, grad_b

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray,
        y_val:   np.ndarray,
    ) -> "RidgeRegression":
        """
        Fit model parameters via mini-batch gradient descent with early stopping.

        Algorithm:
            1. Initialise w ~ 𝒩(0, 0.01²), b = 0
            2. For each epoch:
               a. Shuffle training set
               b. Iterate over mini-batches; update w and b
               c. Evaluate Ridge loss on full train and val sets
               d. Check early stopping criterion
            3. Restore best weights at convergence

        Parameters
        ----------
        X_train : (n_train, p) training features
        y_train : (n_train,)  training targets
        X_val   : (n_val, p)  validation features
        y_val   : (n_val,)    validation targets

        Returns
        -------
        self
        """
        rng = np.random.default_rng(self.random_state)
        n, p = X_train.shape

        # Xavier-style initialisation
        self.w_ = rng.normal(0.0, 0.01, size=p)
        self.b_ = 0.0

        best_val_loss  = np.inf
        best_w = self.w_.copy()
        best_b = self.b_
        no_improve = 0

        logger.info(
            "Training Ridge: n=%d  p=%d  α=%.4f  η=%.5f  max_epochs=%d",
            n, p, self.alpha, self.learning_rate, self.max_epochs,
        )

        for epoch in range(self.max_epochs):
            # ── Shuffle ────────────────────────────────────────────────────
            idx = rng.permutation(n)
            X_s, y_s = X_train[idx], y_train[idx]

            # ── Mini-batch updates ─────────────────────────────────────────
            for start in range(0, n, self.batch_size):
                end  = min(start + self.batch_size, n)
                X_b  = X_s[start:end]
                y_b  = y_s[start:end]

                y_hat_b   = self._predict_raw(X_b)
                residuals = y_hat_b - y_b

                grad_w, grad_b = self._gradients(X_b, residuals)

                self.w_ -= self.learning_rate * grad_w
                self.b_ -= self.learning_rate * grad_b

            # ── Epoch-level evaluation ─────────────────────────────────────
            train_loss = self._ridge_loss(self._predict_raw(X_train) - y_train)
            val_loss   = self._mse_loss(self._predict_raw(X_val)   - y_val)

            self.train_losses_.append(train_loss)
            self.val_losses_.append(val_loss)

            # ── Early Stopping ─────────────────────────────────────────────
            if val_loss < best_val_loss - self.tol:
                best_val_loss  = val_loss
                best_w = self.w_.copy()
                best_b = self.b_
                self.best_epoch_ = epoch
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= self.patience:
                logger.info(
                    "Early stopping at epoch %d  (best_val_loss=%.6f  epoch=%d)",
                    epoch, best_val_loss, self.best_epoch_,
                )
                break

            if epoch % 100 == 0:
                logger.debug(
                    "Epoch %4d | train_loss=%.6f | val_loss=%.6f",
                    epoch, train_loss, val_loss,
                )

        # ── Restore best weights ───────────────────────────────────────────
        self.w_ = best_w
        self.b_ = best_b
        logger.info("Training complete. Best epoch: %d", self.best_epoch_)
        return self

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate continuous predictions ŷ = Xw + b.

        Parameters
        ----------
        X : np.ndarray of shape (n, p)

        Returns
        -------
        np.ndarray of shape (n,)
        """
        if self.w_ is None:
            raise RuntimeError("Model has not been fitted.  Call .fit() first.")
        return self._predict_raw(X)

    # -----------------------------------------------------------------------
    # Serialisation
    # -----------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialise model to pickle."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Model saved → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "RidgeRegression":
        """Deserialise model from pickle."""
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info("Model loaded ← %s", path)
        return model

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------

    @property
    def feature_weights(self) -> np.ndarray:
        """Return the fitted weight vector w_ (copy)."""
        if self.w_ is None:
            raise RuntimeError("Model not fitted.")
        return self.w_.copy()

    def training_history(self) -> Dict[str, List[float]]:
        """Return dict of train/val loss curves."""
        return {
            "train_loss": self.train_losses_,
            "val_loss":   self.val_losses_,
        }