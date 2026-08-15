"""Cross-validation, tuning, and holdout diagnostics."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)

from .data import split_features_target
from .models import MODEL_NAMES, build_model, parameter_grid

SCORING = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1_macro": "f1_macro",
}


def _validate_folds(target: pd.Series, folds: int) -> None:
    if folds < 2:
        raise ValueError("folds must be at least 2")
    smallest_class = int(target.value_counts().min())
    if folds > smallest_class:
        raise ValueError(
            f"folds={folds} exceeds the smallest class size ({smallest_class})"
        )


def compare_models(
    frame: pd.DataFrame,
    *,
    models: Iterable[str] = MODEL_NAMES,
    folds: int = 5,
    repeats: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare pipelines on identical repeated stratified folds."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    features, target = split_features_target(frame)
    _validate_folds(target, folds)
    splitter = RepeatedStratifiedKFold(
        n_splits=folds, n_repeats=repeats, random_state=seed
    )

    rows: list[dict[str, float | str]] = []
    for name in models:
        estimator = build_model(name, seed)
        scores = cross_validate(
            estimator,
            features,
            target,
            cv=splitter,
            scoring=SCORING,
            n_jobs=1,
            error_score="raise",
        )
        rows.append(
            {
                "model": name,
                "accuracy_mean": float(scores["test_accuracy"].mean()),
                "accuracy_std": float(scores["test_accuracy"].std()),
                "balanced_accuracy_mean": float(
                    scores["test_balanced_accuracy"].mean()
                ),
                "f1_macro_mean": float(scores["test_f1_macro"].mean()),
                "fit_time_mean_seconds": float(scores["fit_time"].mean()),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["f1_macro_mean", "accuracy_mean", "model"], ascending=[False, False, True])
        .reset_index(drop=True)
    )


def tune_model(
    frame: pd.DataFrame,
    name: str,
    *,
    folds: int = 5,
    seed: int = 42,
) -> GridSearchCV:
    """Tune one pipeline using stratified CV and macro F1."""
    features, target = split_features_target(frame)
    _validate_folds(target, folds)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    search = GridSearchCV(
        build_model(name, seed),
        parameter_grid(name),
        scoring="f1_macro",
        cv=splitter,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
    return search.fit(features, target)


def evaluate_holdout(
    frame: pd.DataFrame,
    name: str,
    *,
    test_size: float = 0.2,
    seed: int = 42,
    tune: bool = False,
    tuning_folds: int = 5,
) -> dict[str, Any]:
    """Evaluate one model on a stratified holdout and return serializable diagnostics."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    features, target = split_features_target(frame)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        stratify=target,
        random_state=seed,
    )

    estimator = build_model(name, seed)
    best_parameters: dict[str, Any] = {}
    if tune:
        _validate_folds(y_train, tuning_folds)
        splitter = StratifiedKFold(
            n_splits=tuning_folds, shuffle=True, random_state=seed
        )
        search = GridSearchCV(
            estimator,
            parameter_grid(name),
            scoring="f1_macro",
            cv=splitter,
            n_jobs=1,
            refit=True,
            error_score="raise",
        ).fit(x_train, y_train)
        estimator = search.best_estimator_
        best_parameters = search.best_params_
    else:
        estimator.fit(x_train, y_train)

    predictions = estimator.predict(x_test)
    labels = sorted(str(value) for value in target.unique())
    return {
        "model": name,
        "seed": seed,
        "test_size": test_size,
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "tuned": tune,
        "best_parameters": best_parameters,
        "labels": labels,
        "metrics": {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
            "f1_macro": float(f1_score(y_test, predictions, average="macro")),
        },
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
        "classification_report": classification_report(
            y_test, predictions, labels=labels, output_dict=True, zero_division=0
        ),
    }
