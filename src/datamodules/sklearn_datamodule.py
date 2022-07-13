from typing import Tuple, Callable
import numpy as np
from pl_bolts.datamodules import SklearnDataModule
from src.utils import get_logger
import functools

log = get_logger(__name__)


# Parameters that can be passed as kwargs:
# https://pytorch-lightning-bolts.readthedocs.io/en/latest/datamodules_sklearn.html#sklearn-datamodule-class
def create_sklearn_datamodule(dataset: Tuple[np.ndarray, np.ndarray], data_aug: functools.partial = None,
                              *args, **kwargs):
    X, y = dataset
    y = np.reshape(y, (len(y), 1))  # reshape to be (len(y), 1) vector instead of a flat array

    if data_aug:
        log.info(f"Data augmentation function provided {data_aug.__repr__()}")
        log.info(f"Dataset size before augmentation: {len(y)}")
        X, y = data_aug(X, y)
        log.info(f"Dataset size after augmentation: {len(y)}")

    datamodule: SklearnDataModule = SklearnDataModule(X, y, *args, *kwargs)
    log.info(f"SklearnDataModule stats: ")
    log.info(f"\tNumber of Training examples: {len(datamodule.train_dataset)}")
    log.info(f"\tNumber of Validation examples: {len(datamodule.val_dataset)}")
    log.info(f"\tNumber of Testing examples: {len(datamodule.test_dataset)}")

    return datamodule
