from torch import nn
import pytorch_lightning as pl
import more_itertools as mit
from collections import OrderedDict
from utils.get_pytorch_functions import get_loss_function, get_optimizer


# define a Multi-Layer Perceptron
class MLP(pl.LightningModule):
    def __init__(self, layers, loss_function, optimizer, lr):
        super().__init__()
        self.loss_fn = get_loss_function(loss_function)

        # optimizer params
        self.optimizer_fn = get_optimizer(optimizer)
        self.lr = lr

        # define the sequential network
        number_of_trainable_layers = len(layers) - 1
        print(f"number of trainable layers {number_of_trainable_layers}")

        input_output_dims = list(mit.windowed(layers, n=2, step=1))
        print(input_output_dims)

        layer_dict = OrderedDict()
        for i, dims in enumerate(input_output_dims):
            input_dims, output_dims = dims
            layer_dict[f"linear_{i}"] = nn.Linear(in_features=input_dims, out_features=output_dims)

            # don't add relu to final logits layer
            if i != number_of_trainable_layers-1:
                layer_dict[f"relu_{i}"] = nn.ReLU()

        print(layer_dict)

        self.model = nn.Sequential(layer_dict)

    def training_step(self, batch, batch_idx):
        # training_step defines one step of the training loop

        # get one batch of data
        x, y = batch

        # perform forward pass
        y_hat = self.model(x)

        # compute loss
        loss = self.loss_fn(y_hat, y)

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = self.optimizer_fn(self.parameters(), lr=self.lr)
        return optimizer






