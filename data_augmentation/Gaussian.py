import numpy as np

rng = np.random.default_rng(seed=0)


def augment_data(X, y, noise=0.05, iterations=1):
    original_X, original_y = X.copy(), y.copy()

    new_X, new_y = X.copy(), y.copy()
    for _ in range(iterations):
        augmented_X, augmented_y = original_X.copy(), original_y.copy()
        augmented_X += rng.normal(scale=noise, size=augmented_X.shape)
        new_X, new_y = np.vstack((new_X, augmented_X)), np.vstack((new_y, augmented_y))

    return new_X, new_y