from typing import Any, List

import torch
import pytorch_lightning as pl
import more_itertools as mit
from collections import OrderedDict
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import F1Score
from torchmetrics import MaxMetric
from src import utils

log = utils.get_logger(__name__)


# define a Multi-Layer Perceptron
class MLP(pl.LightningModule):
    def __init__(self, layers: list, lr: float = 0.001):
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
            layer_dict[f"linear_{i}"] = torch.nn.Linear(in_features=input_dims, out_features=output_dims)

            # don't add relu to final logits layer
            if i != number_of_trainable_layers-1:
                layer_dict[f"relu_{i}"] = torch.nn.ReLU()

        log.info(layer_dict)

        self.net = torch.nn.Sequential(layer_dict)

        # loss function
        self.criterion = torch.nn.BCEWithLogitsLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        self.train_f1 = F1Score()
        self.val_f1 = F1Score()
        self.test_f1 = F1Score()

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        # y is int64, BCE requires float
        # https://stackoverflow.com/questions/70216222/pytorch-is-throwing-an-error-runtimeerror-result-type-float-cant-be-cast-to-th
        loss = self.criterion(logits, y.float())
        probs = torch.sigmoid(logits)
        preds = probs > 0.5
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        # training_step defines one step of the training loop
        loss, preds, targets = self.step(batch)

        # log train metrics
        acc = self.train_acc(preds, targets)
        f1 = self.train_f1(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1", f1, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_acc(preds, targets)
        f1 = self.val_f1(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", f1, on_step=False, on_epoch=True, prog_bar=True)


        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)


    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_acc(preds, targets)
        f1 = self.test_f1(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)
        self.log("test/f1", f1, on_step=False, on_epoch=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        self.train_acc.reset()
        self.test_acc.reset()
        self.val_acc.reset()

        self.train_f1.reset()
        self.test_f1.reset()
        self.val_f1.reset()


    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.SGD(
            params=self.parameters(),
            lr=self.hparams.lr
        )
