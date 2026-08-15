"""Model registry with preprocessing kept inside scikit-learn pipelines."""

from __future__ import annotations

from collections.abc import Callable

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def _knn(_: int) -> BaseEstimator:
    return Pipeline(
        [("scale", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=5))]
    )


def _logistic(seed: int) -> BaseEstimator:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2_000, random_state=seed)),
        ]
    )


def _svm(seed: int) -> BaseEstimator:
    return Pipeline(
        [("scale", StandardScaler()), ("model", SVC(kernel="rbf", random_state=seed))]
    )


def _forest(seed: int) -> BaseEstimator:
    return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1)


def _naive_bayes(_: int) -> BaseEstimator:
    return GaussianNB()


_BUILDERS: dict[str, Callable[[int], BaseEstimator]] = {
    "gaussian_nb": _naive_bayes,
    "knn": _knn,
    "logistic_regression": _logistic,
    "random_forest": _forest,
    "rbf_svm": _svm,
}
MODEL_NAMES = tuple(_BUILDERS)


_PARAMETER_GRIDS: dict[str, dict[str, list[object]]] = {
    "gaussian_nb": {"var_smoothing": [1e-11, 1e-10, 1e-9, 1e-8, 1e-7]},
    "knn": {
        "model__n_neighbors": [3, 5, 7, 9, 11],
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
    },
    "logistic_regression": {"model__C": [0.1, 0.5, 1.0, 5.0, 10.0]},
    "random_forest": {
        "max_depth": [None, 3, 5],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", None],
    },
    "rbf_svm": {
        "model__C": [0.5, 1.0, 5.0, 10.0],
        "model__gamma": ["scale", 0.01, 0.1, 1.0],
    },
}


def build_model(name: str, seed: int = 42) -> BaseEstimator:
    """Create a fresh estimator by registry name."""
    try:
        return _BUILDERS[name](seed)
    except KeyError as exc:
        raise ValueError(f"Unknown model '{name}'. Choose from: {', '.join(MODEL_NAMES)}") from exc


def parameter_grid(name: str) -> dict[str, list[object]]:
    """Return a defensive copy of the model's compact educational search grid."""
    if name not in _PARAMETER_GRIDS:
        raise ValueError(f"Unknown model '{name}'. Choose from: {', '.join(MODEL_NAMES)}")
    return {key: list(values) for key, values in _PARAMETER_GRIDS[name].items()}
