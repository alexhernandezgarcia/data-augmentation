from typing import Tuple
import numpy as np

rng = np.random.default_rng()


def augment_data(X: np.ndarray, y: np.ndarray, noise: float = 0.05, iterations: int = 1) -> Tuple[np.ndarray,np.ndarray]:

    original_X, original_y = X.copy(), y.copy()

    new_X, new_y = X.copy(), y.copy()
    for _ in range(iterations):
        augmented_X, augmented_y = original_X.copy(), original_y.copy()
        augmented_X += rng.normal(scale=noise, size=augmented_X.shape)
        new_X, new_y = np.vstack((new_X, augmented_X)), np.vstack((new_y, augmented_y))

    return new_X, new_y