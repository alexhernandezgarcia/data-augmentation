from typing import Tuple, List
import numpy as np
from collections import Counter
from sklearn.mixture import GaussianMixture


def augment_data(
        X: np.ndarray, Y: np.ndarray, n_components: int = 2, n_augmentations: int = 1, **kwargs
) -> Tuple[np.ndarray, np.ndarray]:

    if n_augmentations == 0:
        return X, Y

    n_classes = np.sort(np.unique(Y))
    n_samples_in_each_class = Counter(Y.flatten())
    # print(n_samples_in_each_class)
    training_sets = [X[(Y == yi).flatten()] for yi in n_classes]

    models = [GaussianMixture(n_components=n_components, **kwargs).fit(Xi) for Xi in
                   training_sets]

    new_X, new_Y = X.copy(), Y.copy()

    for c in n_classes:
        augmented_X = models[c].sample(n_samples_in_each_class[c]*n_augmentations)[0]
        # print(augmented_X.shape)
        augmented_Y = np.full((n_samples_in_each_class[c]*n_augmentations, 1), fill_value=c, dtype=Y.dtype)
        # print(augmented_X.shape)
        new_X, new_Y = np.vstack((new_X, augmented_X)), np.vstack((new_Y, augmented_Y))

    return new_X, new_Y