import numpy as np
from sklearn.utils import shuffle as shuffle_data


def make_sine_wave(n_samples=500, noise=0.0, shuffle=False, random_state=0):
    c = 3
    num = n_samples
    step = num / (c * 4)
    np.random.seed(random_state)
    x0 = np.linspace(-c * np.pi, c * np.pi, num)
    x1 = np.sin(x0)
    random_noise = np.random.normal(0, noise, num)
    random_noise = np.sign(x1) * np.abs(random_noise)
    x1 = x1 + random_noise
    x0 = x0 + (np.asarray(range(num)) / step) * 0.3
    X = np.column_stack((x0, x1))
    y = np.asarray([int((i / step) % 2) for i in range(len(x0))])

    if shuffle:
        X, y = shuffle_data(X, y, random_state=random_state)

    return X, y
