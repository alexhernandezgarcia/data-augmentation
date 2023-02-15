from typing import Tuple

import numpy
import numpy as np
from sklearn.utils import shuffle as shuffle_data


def make_pinwheel(
    n_samples: int = 500,
    n_features: int = 2,
    n_classes: int = 3,
    noise: float = 0.0,
    shuffle: bool = False,
    random_state: int = 0,
    remove_origin_points: bool = False,
    radius_from_origin: float = 0.03,
) -> Tuple[np.ndarray, np.ndarray]:
    """

    Args:
        n_samples (int): number of points per class. Default 500
        n_features (int): dimensionality of the resultant dataset. Default of 2
        n_classes (int): number of classes. Default 3
        noise (float): random Gaussian noise to add to disperse points. Default 0.0
        shuffle (bool): whether to shuffle the dataset. Default False
        random_state (int): random seed value for reproducible results. Default 0
        remove_origin_points (bool): whether to remove points around origin. Default False
        radius_from_origin (float): radius around the origin from which the points to remove. Default 0.03

    Returns:
        Dataset (Tuple[np.ndarray, np.ndarray]): X, y

    """
    np.random.seed(random_state)
    X = np.zeros((n_samples * n_classes, n_features))
    y = np.zeros(n_samples * n_classes)
    for j in range(n_classes):
        ix = range(n_samples * j, n_samples * (j + 1))
        # radius
        r = np.linspace(0.0, 1, n_samples)
        # theta
        t = np.linspace(j * 4, (j + 1) * 4, n_samples) + (np.random.randn(n_samples) * noise)
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j

    if remove_origin_points:
        x0, y0, radius = 0.0, 0.0, radius_from_origin
        xx, xy = X[:, 0], X[:, 1]
        r = np.sqrt((xx - x0) ** 2 + (xy - y0) ** 2)
        outside = r > radius
        y = y[outside]
        X = X[outside]

    if shuffle:
        X, y = shuffle_data(X, y, random_state=random_state)
    # fig = plt.figure(figsize=(6, 6))
    # plt.scatter(X[:, 0], X[:, 1], c=y, s=40, alpha=0.8)
    # plt.xlim([-1,1])
    # plt.ylim([-1,1])
    return X, y


if __name__ == "__main__":
    make_pinwheel()
