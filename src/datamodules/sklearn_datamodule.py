from typing import Tuple, Callable
import numpy as np
from pl_bolts.datamodules import SklearnDataModule
from pytorch_lightning import LightningDataModule
from src.utils.pylogger import get_pylogger
from sklearn.model_selection import train_test_split

log = get_pylogger(__name__)


def create_sklearn_datamodule(train_dataset: Tuple[np.ndarray, np.ndarray],
                              val_dataset: Tuple[np.ndarray, np.ndarray],
                              test_dataset: Tuple[np.ndarray, np.ndarray],
                              data_aug: Callable = None, *args: object,
                              **kwargs: object) -> LightningDataModule:
    """
    Helper function to create a LightningDataModule for Sklearn datasets.

    # Parameters that can be passed as args and kwargs:
    # https://pytorch-lightning-bolts.readthedocs.io/en/latest/datamodules_sklearn.html#sklearn-datamodule-class
    Args:
        train_dataset (Tuple[np.ndarray, np.ndarray]): a Sklearn dataset, can be instantiated using Hydra
        val_dataset (Tuple[np.ndarray, np.ndarray]): a fixed Sklearn dataset to be used as validation dataset,
                                                     can be initialized through hydra
        test_dataset (Tuple[np.ndarray, np.ndarray]): a fixed Sklearn dataset to be used as test dataset, 
                                                      can be initialized through hydra
        data_aug (Callable): a callable function/class/object that takes in the X (sample points) and Y (labels) and
                             return augmented dataset new_X and new_Y
        *args: arguments for SklearnDataModule (see comment above for full list of args)
        **kwargs: keyword arguments for SklearnDataModule (see comment above for full list of kwargs)

    Returns:
        datamodule (LightningDataModule): a PytorchLightening Datamodule

    """
    X_train, y_train = train_dataset
    y_train = np.reshape(y_train, (len(y_train), 1))  # reshape to be (len(y_train), 1) vector instead of a flat array

    # setup validation set
    X_val, y_val = val_dataset
    y_val = np.reshape(y_val, (len(y_val), 1))

    # setup test dataset
    X_test, y_test = test_dataset
    y_test = np.reshape(y_test, (len(y_test), 1))

    # augment data if a data_aug callable function is provided
    if data_aug:
        log.info(f"Data augmentation function provided {data_aug.__repr__()}")
        log.info(f"Total dataset size before augmentation: {len(y_train) + len(y_val) + len(y_test)}")
        log.info(f"Training set size before augmentation: {len(y_train)}")
        X_train, y_train = data_aug(X_train, y_train)
        log.info(f"Training set size after augmentation: {len(y_train)}")
        log.info(f"Total dataset size after augmentation: {len(y_train) + len(y_val) + len(y_test)}")

    # create SklearnDataModule, which is just a LightningDataModule
    datamodule: SklearnDataModule = SklearnDataModule(X_train, y_train,
                                                      x_val=X_val, y_val=y_val,
                                                      x_test=X_test, y_test=y_test,
                                                      *args, *kwargs)

    # log info about the dataset.
    log.info(f"SklearnDataModule stats: ")
    log.info(f"\tNumber of Training examples: {len(datamodule.train_dataset)}")
    log.info(f"\tNumber of Validation examples: {len(datamodule.val_dataset)}")
    log.info(f"\tNumber of Testing examples: {len(datamodule.test_dataset)}")

    return datamodule
