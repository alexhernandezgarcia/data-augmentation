import numpy as np
from sklearn.utils import shuffle as shuffle_data


def make_pinwheel(n_samples=500, n_features=2, n_classes=3, noise=0.0, shuffle=False, random_state=0):
    """
    n_samples: number of points per class
    n_features: dimensionality
    n_classes: number of classes
    noise: random Gaussian noise to add to disperse points
    """
    np.random.seed(random_state)
    X = np.zeros((n_samples*n_classes, n_features))
    y = np.zeros(n_samples*n_classes)
    for j in range(n_classes):
        ix = range(n_samples*j, n_samples*(j+1))
        # radius
        r = np.linspace(0.0,1,n_samples)
        # theta
        t = np.linspace(j*4,(j+1)*4,n_samples) + (np.random.randn(n_samples) * noise)
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j

    if shuffle:
        X, y = shuffle_data(X, y, random_state=random_state)
    # fig = plt.figure(figsize=(6, 6))
    # plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu, alpha=0.8)
    # plt.xlim([-1,1])
    # plt.ylim([-1,1])
    return X, y

if __name__ == '__main__':
    make_pinwheel()