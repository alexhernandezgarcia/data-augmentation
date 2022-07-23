from typing import Tuple, Callable
import numpy as np
from pl_bolts.datamodules import SklearnDataModule
from pytorch_lightning import LightningDataModule
from src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)



def create_sklearn_datamodule(dataset: Tuple[np.ndarray, np.ndarray], data_aug: Callable = None,
                              *args: object, **kwargs: object) -> LightningDataModule:
    """
    Helper function to create a LightningDataModule for Sklearn datasets.

    # Parameters that can be passed as args and kwargs:
    # https://pytorch-lightning-bolts.readthedocs.io/en/latest/datamodules_sklearn.html#sklearn-datamodule-class
    Args:
        dataset (Tuple[np.ndarray, np.ndarray]): a Sklearn dataset, can be instantiated using Hydra
        data_aug (Callable): a callable function/class/object that takes in the X (sample points) and Y (labels) and
                             return augmented dataset new_X and new_Y
        *args: arguments for SklearnDataModule (see comment above for full list of args)
        **kwargs: keyword arguments for SklearnDataModule (see comment above for full list of kwargs)

    Returns:
        datamodule (LightningDataModule): a PytorchLightening Datamodule

    """
    X, y = dataset
    y = np.reshape(y, (len(y), 1))  # reshape to be (len(y), 1) vector instead of a flat array

    # augment data if a data_aug callable function is provided
    if data_aug:
        log.info(f"Data augmentation function provided {data_aug.__repr__()}")
        log.info(f"Dataset size before augmentation: {len(y)}")
        X, y = data_aug(X, y)
        log.info(f"Dataset size after augmentation: {len(y)}")

    # create SklearnDataModule, which is just a LightningDataModule
    datamodule: SklearnDataModule = SklearnDataModule(X, y, *args, *kwargs)

    # log info about the dataset.
    log.info(f"SklearnDataModule stats: ")
    log.info(f"\tNumber of Training examples: {len(datamodule.train_dataset)}")
    log.info(f"\tNumber of Validation examples: {len(datamodule.val_dataset)}")
    log.info(f"\tNumber of Testing examples: {len(datamodule.test_dataset)}")

    return datamodule
