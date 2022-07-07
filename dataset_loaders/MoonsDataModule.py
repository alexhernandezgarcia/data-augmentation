from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from sklearn import datasets
import pytorch_lightning as pl
from typing import Optional
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from typing import Callable


class MoonsDataModule(pl.LightningDataModule):
    def __init__(self, number_of_samples: int, noise: float, shuffle: bool = True, val_split: float = 0.1,
                 batch_size: int = 32, augmentation_function: Callable = None):

        super().__init__()
        self.number_of_samples = number_of_samples
        self.noise = noise
        self.shuffle = shuffle
        self.val_split = val_split
        self.batch_size = batch_size
        self.augmentation_function = augmentation_function

    def setup(self, stage: Optional[str] = None):

        # generate moons dataset
        moons = datasets.make_moons(n_samples=self.number_of_samples, shuffle=self.shuffle,
                                    noise=self.noise)

        # Assign Train/val split(s) for use in Dataloaders
        if stage in (None, "fit"):
            X, y = moons
            y = np.reshape(y, (len(y), 1))  # reshape to be (len(y), 1) vector instead of a flat array

            # augment dataset
            if self.augmentation_function:
                X, y = self.augmentation_function(X, y)

            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.val_split, random_state=42)

            # Convert to torch.tensor:
            X_train_t = torch.from_numpy(X_train).to(torch.float32)
            y_train_t = torch.from_numpy(y_train).to(torch.float32)
            X_val_t = torch.from_numpy(X_val).to(torch.float32)
            y_val_t = torch.from_numpy(y_val).to(torch.float32)

            # And make TensorDataset's from them:
            self.train_dataset = TensorDataset(X_train_t, y_train_t)
            self.val_dataset = TensorDataset(X_val_t, y_val_t)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)


def make_multiple_moons_dataloader(base_num_of_samples, noise, shuffle, val_split, batch_size, augmentation_function):
    # create a LighteningDataLoader from each dataset
    moons_dataloaders = {}
    for sample_size in base_num_of_samples:
        moons_dataloaders[sample_size] = MoonsDataModule(sample_size, noise, shuffle, val_split, batch_size, augmentation_function)

    return moons_dataloaders















