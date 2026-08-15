# Methodology

## What is measured

The comparison command evaluates five classical classifiers on the same
repeated stratified folds. Every scale-sensitive estimator keeps
`StandardScaler` inside its scikit-learn `Pipeline`, so preprocessing is fitted
only on each training fold.

The reported metrics are accuracy, balanced accuracy, and macro F1. Macro F1
gives each species equal weight and is used as the primary sort key.

## Reproducibility

- All splitters and stochastic estimators receive an explicit seed.
- The default benchmark uses 5 folds repeated 3 times.
- Model comparison and holdout evaluation are separate commands.
- Machine-readable CSV and JSON outputs can be committed for auditability.

## Tuning and holdout evaluation

`evaluate --tune` splits the data first, then tunes only on the training
portion using stratified cross-validation. The untouched holdout is used once
for confusion-matrix and classification-report diagnostics. This avoids using
holdout labels during hyperparameter selection.

## Limitations

Iris is a small, clean teaching dataset. High scores do not imply that the same
models or hyperparameters will generalize to noisy, imbalanced, drifting, or
high-dimensional production data. The benchmark is intended to demonstrate a
sound evaluation workflow, not to claim a novel model.
