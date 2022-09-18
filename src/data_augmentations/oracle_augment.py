from typing import Tuple, Dict
import numpy as np


def augment_data(
    X: np.ndarray, Y: np.ndarray, X_oracle: np.ndarray, Y_oracle: np.ndarray, max_d: float = 0.1, lmd: float = 0.05, n_augmentations: int = 1, **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment data function takes in a SkLearn point dataset (moons, circles, blobs etc.) and creates a new dataset.
    The new dataset is a combination of the original dataset + the original dataset augmented using a noisy oracle.
    The amount of new augmented data added depends on the number of `n_augmentations` parameter
    Args:
        X (np.ndarray): a Sklearn dataset samples
        Y (np.ndarray): a numpy array with integer values denoting class labels for all data points in X
        X_oracle (np.ndarray): a dense dataset sampled from the same distribution as X.
        Y_oracle (np.ndarray): a numpy array with integer values denoting class labels for all data points in X_oracle
        max_d (float): Controls the amount of Gaussian noise to add to each sample point in SKlearn data set.
        lmd (float): Controls the amount of Gaussian noise used to penalize the oracle. Higher the lmd the more distorted the new points. Default 0.05
        n_augmentations (int): the number of time to run the original dataset through the augmentation process to create
                          new data. Default 1
        **kwargs: keyword arguments for handling mismatched arguments across distinct augmentation functions

    Returns:
        new_X (np.ndarray): augmented dataset with is a combination of original dataset + new augmented dataset
        new_Y (np.ndarray): array of class labels for each sample point in new_X

    """

    new_X, new_Y = X.copy(), Y.copy()
    for X_sample, Y_sample in zip(X, Y):
        
        for _ in range(n_augmentations):
            
            ## Sample Distance from Selected Point
            dx_1 = np.random.normal(scale=max_d)
            dx_2 = np.random.normal(scale=max_d)

            ## Synthetic Data dx_1, dx_2 away from original sample
            X_syn = np.array([X_sample[0] + dx_1, X_sample[1] + dx_2])

            ## Nearest Point Available in Data-Bank
            X_oracle_cls, Y_oracle_cls = X_oracle[Y_oracle == Y_sample], Y_oracle[Y_oracle == Y_sample]
            diff = np.linalg.norm(X_syn - X_oracle_cls, ord=2, axis=1) ## Only evaluate norm over samples of same class in the data-bank
            nearest_idx = np.argmin(diff)
            X_near = X_oracle_cls[nearest_idx]
            Y_near = np.array([Y_oracle_cls[nearest_idx]]).reshape(-1, 1)
            assert Y_sample == Y_near

            ## Penalizing the nearest neighbour based on distance
            X_near_noisy = np.array([X_near[0] + np.random.normal(scale= lmd * np.exp(dx_1)), X_near[1] + np.random.normal(scale= lmd * np.exp(dx_2))]).reshape(-1, 2)

            new_X, new_Y = np.vstack((new_X, X_near_noisy)), np.vstack((new_Y, Y_near))

    return new_X, new_Y
