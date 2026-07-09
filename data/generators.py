"""
Synthetic primary/candidate models and traffic generator.

Two classifiers trained on the same underlying data-generating process, but
one is deliberately given a small real improvement (more trees, slightly
better features) so the benchmark has a genuine, measurable signal instead
of two identical models with no story to tell.
"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def make_dataset(n_samples: int, n_features: int = 12, n_classes: int = 3, seed: int = 42):
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features, n_informative=8,
        n_redundant=2, n_classes=n_classes, n_clusters_per_class=2,
        class_sep=1.1, flip_y=0.03, random_state=seed,
    )
    return X, y


def train_primary_model(X_train, y_train, seed: int = 42):
    """The incumbent production model: a modest logistic regression."""
    model = LogisticRegression(max_iter=1000, random_state=seed)
    model.fit(X_train, y_train)
    return model


def train_candidate_model(X_train, y_train, seed: int = 42, n_estimators: int = 150):
    """The candidate: a random forest, genuinely stronger on this data."""
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=8, random_state=seed, n_jobs=1,
    )
    model.fit(X_train, y_train)
    return model


def train_buggy_candidate_model(X_train, y_train, seed: int = 42):
    """
    A candidate with a genuine, realistic defect: a column-order mismatch
    between the feature pipeline the model was trained on and the one
    serving it in production. This is one of the most common real-world
    causes of a silently broken model. It never raises an error, since the
    shapes and dtypes match, but the model is reading each feature's value
    into the wrong slot.
    """
    rng = np.random.RandomState(seed)
    n_features = X_train.shape[1]
    perm = rng.permutation(n_features)
    while np.any(perm == np.arange(n_features)):
        perm = rng.permutation(n_features)

    model = LogisticRegression(max_iter=1000, random_state=seed)
    model.fit(X_train, y_train)

    class ColumnMismatchWrapper:
        """Trained correctly; served with columns silently reordered."""
        def predict_proba(self, X):
            return model.predict_proba(X[:, perm])

    return ColumnMismatchWrapper()
