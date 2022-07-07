from easydict import EasyDict

import data_augmentation.Gaussian
from models.MLP import MLP
from dataset_loaders.MoonsDataModule import make_multiple_moons_dataloader
from utils.parse_config import get_config_from_yaml
import pytorch_lightning as pl

if __name__ == "__main__":
    # read yaml file
    config = get_config_from_yaml("config.yaml")
    print(config)
    print(config.dataset.batch_size)

    mlp = MLP(layers=config.model.layers,
              loss_function=config.train.loss_func,
              optimizer=config.train.optimizer,
              lr=config.train.lr)

    print(mlp.summarize(max_depth=5))

    # create augmentation function
    aug_function = lambda x, y: data_augmentation.Gaussian.augment_data(x, y, noise=config.augmentation.noise,
                                                                        iterations=config.augmentation.num_of_iterations)

    moons_dls = make_multiple_moons_dataloader(base_num_of_samples=config.dataset.base_num_samples,
                                               noise=config.dataset.noise,
                                               shuffle=config.dataset.shuffle,
                                               val_split=config.dataset.val_split,
                                               batch_size=config.dataset.batch_size,
                                               augmentation_function=aug_function)

    for dataset_size, datamodule in moons_dls.items():
        trainer = pl.Trainer(max_epochs=config.train.epochs, log_every_n_steps=10)
        model = MLP(layers=config.model.layers, loss_function=config.train.loss_func,
                    optimizer=config.train.optimizer,
                    lr=config.train.lr)

        trainer.fit(model, datamodule=datamodule)
