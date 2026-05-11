"""
src/models/baseline.py
=======================
Persistence Baseline (Naïve Forecaster)

Mathematical definition
-----------------------
The persistence model assumes the future equals the present:

    ŷ_t = y_{t−1}

This is the simplest possible non-trivial forecaster and acts as the
lower bound for meaningful model comparison.

Improvement metric:
    RMSE_improvement = (1 − RMSE_model / RMSE_baseline) × 100%

A positive percentage signals that the learned model beats the naïve forecast.
"""

from __future__ import annotations

import numpy as np


class PersistenceBaseline:
    """
    Naïve one-step persistence model: ŷ_t = y_{t−1}.

    Parameters
    ----------
    None

    Notes
    -----
    The model is "fit" by simply retaining the training set targets
    so the last training value can seed test predictions.
    """

    def __init__(self) -> None:
        self._last_train_value: float | None = None

    def fit(self, y_train: np.ndarray) -> "PersistenceBaseline":
        """Store last training value for potential multi-step use."""
        self._last_train_value = float(y_train[-1])
        return self

    def predict(self, y: np.ndarray) -> np.ndarray:
        """
        Return the 1-lag shifted array.

        For a sequence  [y₀, y₁, …, y_{n−1}]  the prediction is:
            [y₀, y₀, y₁, …, y_{n−2}]
        i.e. the first prediction equals y₀ (using the last training value
        if seed is available, otherwise y₀ from the input).

        Parameters
        ----------
        y : np.ndarray of shape (n,)

        Returns
        -------
        np.ndarray of shape (n,)
        """
        if self._last_train_value is not None:
            shifted = np.empty_like(y)
            shifted[0]  = self._last_train_value
            shifted[1:] = y[:-1]
        else:
            shifted = np.roll(y, 1)
            shifted[0] = y[0]
        return shifted