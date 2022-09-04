from typing import Tuple

import numpy as np


def augment_data(
    X: np.ndarray, Y: np.ndarray, noise: float = 0.05, n_augmentations: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment data function takes in a SkLearn point dataset (moons, circles, blobs etc.) and creates a new dataset.
    The new dataset is a combination of the original dataset + the original dataset distorted by random Gaussian noise.
    The amount of new Gaussian distorted data added depends on the number of `iterations` parameter
    Args:
        X (np.ndarray): a Sklearn dataset samples
        Y (np.ndarray): a numpy array with integer values denoting class labels for all data points in X
        noise (float): the amount of Gaussian noise to add to each sample point in X. Higher the noise the more
                       distorted the new points. Default 0.05
        n_augmentations (int): the number of time to run the original dataset through the augmentation process to create
                          new data. Default 1

    Returns:
        new_X (np.ndarray): augmented dataset with is a combination of original dataset + new augmented dataset
        new_Y (np.ndarray): array of class labels for each sample point in new_X

    """

    # instantiate a random number generator.
    # seed value can be set in the experiment configs globally
    rng = np.random.default_rng()

    new_X, new_Y = X.copy(), Y.copy()

    for _ in range(n_augmentations):
        augmented_X, augmented_Y = X.copy(), Y.copy()
        augmented_X += rng.normal(scale=noise, size=augmented_X.shape)
        new_X, new_Y = np.vstack((new_X, augmented_X)), np.vstack((new_Y, augmented_Y))

    return new_X, new_Y
