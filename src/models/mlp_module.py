from collections import OrderedDict
from typing import Any, List, Literal

import more_itertools as mit
import pytorch_lightning as pl
import torch
from torchmetrics import F1Score, MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


# define a Multi-Layer Perceptron
class MLPLitModule(pl.LightningModule):
    def __init__(
        self,
        layers: List[int],
        scheduler: torch.optim.lr_scheduler,
        optimizer: torch.optim.Optimizer,
        task: Literal["binary", "multiclass"] = "binary",
        num_classes: int = 2,
    ) -> None:
        """Constructor for a multi-layer perceptron style network with ReLU activations in hidden
        layers and Stochastic Gradient Descent (SGD) as the optimizer.

        Args:
            layers (List[int]): a list of integer values which define the depth and width of the feed-forward
                                dense network.

            scheduler (torch.optim.lr_scheduler): a learning rate scheduler
            optimizer (torch.optim.Optimizer): optimizer algorithm to use
            task (str): defines the task either 'binary' or 'multiclass', used for setting up metric calculations
                        such as Accuracy and F1. Default task='binary'.
            num_classes (int): number of classes in the dataset. This parameter is used when task is 'multiclass'
                               Default num_classes=2.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # define the sequential network
        number_of_trainable_layers = len(layers) - 1
        log.info(f"number of trainable layers {number_of_trainable_layers}")

        input_output_dims = list(mit.windowed(layers, n=2, step=1))
        log.info(input_output_dims)

        layer_dict = OrderedDict()
        for i, dims in enumerate(input_output_dims):
            input_dims, output_dims = dims
            layer_dict[f"linear_{i}"] = torch.nn.Linear(
                in_features=input_dims, out_features=output_dims
            )

            # don't add relu to final logits layer
            if i != number_of_trainable_layers - 1:
                layer_dict[f"relu_{i}"] = torch.nn.ReLU()

        log.info(layer_dict)

        self.net = torch.nn.Sequential(layer_dict)

        # save the hyper-param 'task'  and 'num_classes' for later use
        self.task = task
        self.num_classes = num_classes

        print(f"Training task set as: {self.task}")
        print(f"Number of classes set as: {self.num_classes}")

        # loss function
        if self.task == "binary":
            self.criterion = torch.nn.BCEWithLogitsLoss()
        elif self.task == "multiclass":
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError(
                f"{task} not found! Value task can be one of the following: 'binary' or 'multiclass'"
            )

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy(task=self.task, num_classes=self.num_classes)
        self.val_acc = Accuracy(task=self.task, num_classes=self.num_classes)
        self.test_acc = Accuracy(task=self.task, num_classes=self.num_classes)

        self.train_f1 = F1Score(task=self.task, num_classes=self.num_classes)
        self.val_f1 = F1Score(task=self.task, num_classes=self.num_classes)
        self.test_f1 = F1Score(task=self.task, num_classes=self.num_classes)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()

    def model_step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)

        if self.task == "binary":
            # y is int64, BCE requires float
            # https://stackoverflow.com/questions/70216222/pytorch-is-throwing-an-error-runtimeerror-result-type-float-cant-be-cast-to-th
            loss = self.criterion(logits, y.float())
            probs = torch.sigmoid(logits)
        else:
            y = torch.squeeze(y).to(torch.long)
            loss = self.criterion(logits, y)
            probs = logits.softmax(dim=-1)

        return loss, probs, y

    def training_step(self, batch: Any, batch_idx: int):
        # training_step defines one step of the training loop
        loss, probs, targets = self.model_step(batch)

        # update  log train metrics
        self.train_loss(loss)
        self.train_acc(probs, targets)
        self.train_f1(probs, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "probs": probs, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`

        # Warning: when overriding `training_epoch_end()`, lightning accumulates outputs from all batches of the epoch
        # this may not be an issue when training on small datasets
        # but on larger datasets/models it's easy to run into out-of-memory errors

        # consider detaching tensors before returning them from `training_step()`
        # or using `on_train_epoch_end()` instead which doesn't accumulate outputs

        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, probs, targets = self.model_step(batch)

        # PL will convert probs to binary by using the >= 0.5 threshold
        # log val metrics
        self.val_loss(loss)
        self.val_acc(probs, targets)
        self.val_f1(probs, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "probs": probs, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.val_acc_best(acc)  # update best so far val acc

        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, probs, targets = self.model_step(batch)

        # PL will convert probs to binary by using the >= 0.5 threshold
        # log test metrics
        self.test_loss(loss)
        self.test_acc(probs, targets)
        self.test_f1(probs, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "probs": probs, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "mlp.yaml")
    _ = hydra.utils.instantiate(cfg)
