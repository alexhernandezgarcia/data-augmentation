from typing import Tuple, List
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from collections import Counter


def augment_data(
        X: np.ndarray, Y: np.ndarray, n_augmentations: int = 1, bandwidths=None, **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    if n_augmentations == 0:
        return X, Y

    n_classes = np.sort(np.unique(Y))
    n_samples_in_each_class = Counter(Y.flatten())
    training_sets = [X[(Y == yi).flatten()] for yi in n_classes]

    # use grid search cross-validation to optimize the bandwidth
    params = {"bandwidth": bandwidths if bandwidths else np.logspace(-1, 1, 500)}

    # models = [GridSearchCV(KernelDensity(), params, cv=LeaveOneOut().get_n_splits(Xi)).fit(Xi) for
    #                               Xi in training_sets]
    models = [GridSearchCV(KernelDensity(), params, cv=5).fit(Xi) for
     Xi in training_sets]

    best_models = [kde_model.best_estimator_ for kde_model in models]

    new_X, new_Y = X.copy(), Y.copy()

    for c in n_classes:
        augmented_X = best_models[c].sample(n_samples_in_each_class[c] * n_augmentations)
        augmented_Y = np.full((n_samples_in_each_class[c] * n_augmentations, 1), fill_value=c, dtype=Y.dtype)
        new_X, new_Y = np.vstack((new_X, augmented_X)), np.vstack((new_Y, augmented_Y))

    return new_X, new_Y

