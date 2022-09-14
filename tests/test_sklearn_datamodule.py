from pathlib import Path

import hydra.utils
import pytest
import torch
from sklearn import datasets
import numpy as np
from src.datamodules.sklearn_datamodule import SklearnDataModule
from hydra import initialize, compose

def find_intersecting_rows(arr1: np.ndarray, arr2: np.ndarray):
    nrows, ncols = arr1.shape
    dtype = {'names': ['f{}'.format(i) for i in range(ncols)],
             'formats': ncols * [arr1.dtype]}

    new_arr = np.intersect1d(arr1.view(dtype), arr2.view(dtype))

    # This last bit is optional if you're okay with "new_arr" being a structured array...
    new_arr = new_arr.view(arr1.dtype).reshape(-1, ncols)

    return new_arr


@pytest.mark.parametrize("train_dataset, val_dataset, test_dataset",
                         [(datasets.make_moons(n_samples=20, shuffle=True, noise=0.0005, random_state=1234),
                           datasets.make_moons(n_samples=1000, shuffle=True, noise=0.0005, random_state=1111),
                           datasets.make_moons(n_samples=1000, shuffle=True, noise=0.0005, random_state=2222))
                          ]
                         )
def test_sklearn_datamodule(train_dataset, val_dataset, test_dataset):
    x_train, y_train = train_dataset
    x_val, y_val = val_dataset
    x_test, y_test = test_dataset

    # combine x and y into one dataset
    combined_train = np.hstack((x_train, np.reshape(y_train, (len(y_train), 1))))
    combined_val = np.hstack((x_val, np.reshape(y_val, (len(y_val), 1))))
    combined_test = np.hstack((x_test, np.reshape(y_test, (len(y_test), 1))))

    # assert all unique values
    assert len(find_intersecting_rows(combined_train, combined_val)) == 0
    assert len(find_intersecting_rows(combined_train, combined_test)) == 0
    assert len(find_intersecting_rows(combined_val, combined_test)) == 0

    with initialize(version_base="1.2", config_path="../configs/datamodule"):
        cfg = compose(config_name="moons.yaml")
        dm = hydra.utils.instantiate(cfg)

        assert dm
        combined_train = np.hstack((dm.train_dataset.X, np.reshape(dm.train_dataset.Y, (len(dm.train_dataset.Y), 1))))
        combined_val = np.hstack((dm.val_dataset.X, np.reshape(dm.val_dataset.Y, (len(dm.val_dataset.Y), 1))))
        combined_test = np.hstack((dm.test_dataset.X, np.reshape(dm.test_dataset.Y, (len(dm.test_dataset.Y), 1))))
        assert len(find_intersecting_rows(combined_train, combined_val)) == 0
        assert len(find_intersecting_rows(combined_train, combined_test)) == 0
        assert len(find_intersecting_rows(combined_val, combined_test)) == 0

        batch_size = dm.batch_size
        batch = next(iter(dm.train_dataloader()))
        x, y = batch
        assert len(x) == batch_size
        assert len(y) == batch_size
        assert x.dtype == torch.float32
        assert y.dtype == torch.int64
