import subprocess
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score


def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """Safely get Weights&Biases logger from Trainer."""

    if trainer.fast_dev_run:
        raise Exception(
            "Cannot use wandb callbacks since pytorch lightning disables loggers in `fast_dev_run=true` mode."
        )

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    raise Exception(
        "You are using wandb related callback, but WandbLogger was not found for some reason..."
    )


class WatchModel(Callback):
    """Make wandb watch model at the beginning of the run."""

    def __init__(self, log: str = "gradients", log_freq: int = 100, log_graph=True):
        self.log = log
        self.log_freq = log_freq
        self.log_graph = log_graph

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        logger.watch(model=trainer.model, log=self.log, log_freq=self.log_freq, log_graph=self.log_graph)


class UploadCodeAsArtifact(Callback):
    """Upload all code files to wandb as an artifact, at the beginning of the run."""

    def __init__(self, code_dir: str, use_git: bool = True):
        """
        Args:
            code_dir: the code directory
            use_git: if using git, then upload all files that are not ignored by git.
            if not using git, then upload all '*.py' file
        """
        self.code_dir = code_dir
        self.use_git = use_git

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        code = wandb.Artifact("project-source", type="code")

        if self.use_git:
            # get .git folder path
            git_dir_path = Path(
                subprocess.check_output(["git", "rev-parse", "--git-dir"]).strip().decode("utf8")
            ).resolve()

            for path in Path(self.code_dir).resolve().rglob("*"):

                # don't upload files ignored by git
                # https://alexwlchan.net/2020/11/a-python-function-to-ignore-a-path-with-git-info-exclude/
                command = ["git", "check-ignore", "-q", str(path)]
                not_ignored = subprocess.run(command).returncode == 1

                # don't upload files from .git folder
                not_git = not str(path).startswith(str(git_dir_path))

                if path.is_file() and not_git and not_ignored:
                    code.add_file(str(path), name=str(path.relative_to(self.code_dir)))

        else:
            for path in Path(self.code_dir).resolve().rglob("*.py"):
                code.add_file(str(path), name=str(path.relative_to(self.code_dir)))

        experiment.log_artifact(code)

class UploadCheckpointsAsArtifact(Callback):
    """Upload checkpoints to wandb as an artifact, at the end of run."""

    def __init__(self, ckpt_dir: str = "checkpoints/", upload_best_only: bool = False):
        self.ckpt_dir = ckpt_dir
        self.upload_best_only = upload_best_only

    @rank_zero_only
    def on_keyboard_interrupt(self, trainer, pl_module):
        self.on_train_end(trainer, pl_module)

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        ckpts = wandb.Artifact("experiment-ckpts", type="checkpoints")

        if self.upload_best_only:
            ckpts.add_file(trainer.checkpoint_callback.best_model_path)
        else:
            for path in Path(self.ckpt_dir).rglob("*.ckpt"):
                ckpts.add_file(str(path))

        experiment.log_artifact(ckpts)

class LogConfusionMatrix(Callback):
    """Generate confusion matrix every epoch and send it to wandb.
    Expects validation step to return predictions and targets.
    """

    def __init__(self):
        self.preds = []
        self.targets = []
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module) -> None:
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Gather data from single batch."""
        if self.ready:
            self.preds.append(outputs["preds"])
            self.targets.append(outputs["targets"])

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate confusion matrix."""
        if self.ready:
            logger = get_wandb_logger(trainer)
            experiment = logger.experiment

            preds = torch.cat(self.preds).cpu().numpy()
            targets = torch.cat(self.targets).cpu().numpy()

            confusion_matrix = metrics.confusion_matrix(y_true=targets, y_pred=preds)

            # set figure size
            plt.figure(figsize=(14, 8))

            # set labels size
            sn.set(font_scale=1.4)

            # set font size
            sn.heatmap(confusion_matrix, annot=True, annot_kws={"size": 8}, fmt="g")

            # names should be unique or else charts from different experiments in wandb will overlap
            experiment.log({f"confusion_matrix/{experiment.name}": wandb.Image(plt)}, commit=False)

            # according to wandb docs this should also work but it crashes
            # experiment.log(f{"confusion_matrix/{experiment.name}": plt})

            # reset plot
            plt.clf()

            self.preds.clear()
            self.targets.clear()


class LogF1PrecRecHeatmap(Callback):
    """Generate f1, precision, recall heatmap every epoch and send it to wandb.
    Expects validation step to return predictions and targets.
    """

    def __init__(self, class_names: List[str] = None):
        self.preds = []
        self.targets = []
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Gather data from single batch."""
        if self.ready:
            self.preds.append(outputs["preds"])
            self.targets.append(outputs["targets"])

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate f1, precision and recall heatmap."""
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment

            preds = torch.cat(self.preds).cpu().numpy()
            targets = torch.cat(self.targets).cpu().numpy()
            f1 = f1_score(targets, preds, average=None)
            r = recall_score(targets, preds, average=None)
            p = precision_score(targets, preds, average=None)
            data = [f1, p, r]

            # set figure size
            plt.figure(figsize=(14, 3))

            # set labels size
            sn.set(font_scale=1.2)

            # set font size
            sn.heatmap(
                data,
                annot=True,
                annot_kws={"size": 10},
                fmt=".3f",
                yticklabels=["F1", "Precision", "Recall"],
            )

            # names should be uniqe or else charts from different experiments in wandb will overlap
            experiment.log({f"f1_p_r_heatmap/{experiment.name}": wandb.Image(plt)}, commit=False)

            # reset plot
            plt.clf()

            self.preds.clear()
            self.targets.clear()


class LogImagePredictions(Callback):
    """Logs a validation batch and their predictions to wandb.
    Example adapted from:
        https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY
    """

    def __init__(self, num_samples: int = 8):
        super().__init__()
        self.num_samples = num_samples
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment

            # get a validation batch from the validation dat loader
            val_samples = next(iter(trainer.datamodule.val_dataloader()))
            val_imgs, val_labels = val_samples

            # run the batch through the network
            val_imgs = val_imgs.to(device=pl_module.device)
            logits = pl_module(val_imgs)
            preds = torch.argmax(logits, dim=-1)

            # log the images as wandb Image
            experiment.log(
                {
                    f"Images/{experiment.name}": [
                        wandb.Image(x, caption=f"Pred:{pred}, Label:{y}")
                        for x, pred, y in zip(
                            val_imgs[: self.num_samples],
                            preds[: self.num_samples],
                            val_labels[: self.num_samples],
                        )
                    ]
                }
            )


class LogDecisionBoundary(Callback):
    """
    Logs decision boundary on the validation set. The decision boundary itself is saved as decision_boundary.png
    under the logs directory for the experiment
    """

    def __init__(self, dirpath: str):
        """
        Constructor for LogDecisionBoundary callback
        Args:
            dirpath (str): path where to save the decision boundary
        """
        self.dirpath = dirpath

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """
               Callback runs when training ends
               Args:
                   trainer: lightning trainer
                   pl_module: lightning module
        """

        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        # get validation data
        val_data = trainer.datamodule.val_dataloader().dataset
        valX, valY = val_data.X, val_data.Y

        pl_module.eval()  # put model in eval mode
        self._show_separation(model=pl_module, experiment_logger=experiment, X=valX, y=valY)
        pl_module.train()  # put model back to train mode

    def _show_separation(self, model: pl.LightningModule, experiment_logger: WandbLogger.experiment,
                         X: np.ndarray, y: np.ndarray, save: bool = True):
        """
        Plots and logs decision boundary for a model and dataset (X, y)

        Args:
            model (pl.LightningModule):  lightning module
            experiment_logger (WandbLogger.experiment):  Wandb experiment run logger
            X (np.ndarray): Dataset samples
            y (np.ndarray) : Dataset labels
            save (bool): whether to save decision boundary plot or not.
        """
        sn.set(style="white")

        xx, yy = np.mgrid[-1.5:2.5:.01, -1.:1.5:.01]
        grid = np.c_[xx.ravel(), yy.ravel()]
        batch = torch.from_numpy(grid).type(torch.float32)
        with torch.no_grad():
            probs = torch.sigmoid(model(batch).reshape(xx.shape))
            probs = probs.numpy().reshape(xx.shape)

        f, ax = plt.subplots(figsize=(16, 10))

        ax.set_title("Decision boundary", fontsize=14)
        contour = ax.contourf(xx, yy, probs, 25, cmap="RdBu",
                              vmin=0, vmax=1)
        #     ax_c = f.colorbar(contour)
        #     ax_c.set_label("$P(y = 1)$")
        #     ax_c.set_ticks([0, .25, .5, .75, 1])

        ax.scatter(X[:, 0], X[:, 1], c=y, s=50,
                   cmap="RdBu", vmin=-.2, vmax=1.2,
                   edgecolor="white", linewidth=1)

        ax.set(xlabel="$X_1$", ylabel="$X_2$")
        if save:
            plt.savefig(self.dirpath+"/decision_boundary.png")

        experiment_logger.log({f"decision_boundary/{experiment_logger.name}": wandb.Image(plt)}, commit=False)
        plt.clf()


class LogWeightBiasDistribution(Callback):
    """
    Logs weights and bias distribution. The distribution plots are saved user the distribution plots directory under
    the logs directory
    """

    def __init__(self, dirpath: str):
        """
        Constructor for LogDecisionBoundary callback
        Args:
            dirpath (str): path where to save the decision boundary
        """
        self.dirpath = dirpath

        # subdirectory where plots of before and after training can be found
        self._plots_directory = "distribution_plots"

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """
        Callback runs when training is about to starts
        Args:
            trainer: lightning trainer
            pl_module: lightning module
        """
        # get the wandb logger
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        # get parameters and their names and send them to plot_distribution
        for name, param in pl_module.named_parameters():
            # print(name, param)
            self.plot_distribution(name, param.data.numpy().ravel(), stage="before", experiment_logger=experiment)

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """
        Callback runs when training ends
        Args:
            trainer: lightning trainer
            pl_module: lightning module
        """
        # get the wandb logger
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        # get parameters and their names and send them to plot_distribution
        for name, param in pl_module.named_parameters():
            # print(name, param)
            self.plot_distribution(name, param.data.numpy().ravel(), stage="after", experiment_logger=experiment)

    def plot_distribution(self, name: str, data: np.ndarray, stage: str, experiment_logger: WandbLogger.experiment) -> None:
        """
        Plots the distribution of data using Seaborn KDE plot, the plot is saved under the directory given by
        self_plots_directory under the log directory for the run
        Args:
            name (str) : parameter name
            data (np.ndarray) : flat numpy.ndarray
            stage (str) : the stage of the callback either "before" or "after" training
            experiment_logger (WandbLogger.experiment): the wandb logger object
        """
        # set figure size
        plt.figure(figsize=(16, 10))
        # set font size
        plt.rcParams.update({'font.size': 22})
        plt.title(name)
        sn.kdeplot(data=data, shade=True, color='red' if "weight" in name else 'blue')
        plt.xlabel("Weight values")

        # create path
        path = Path(self.dirpath)
        path = path / self._plots_directory / stage
        path.mkdir(parents=True, exist_ok=True)
        plt.savefig(path / (name + ".png"))

        experiment_logger.log({f'parameter_values_{stage}_training/{name}': wandb.Image(plt)})

        # column_name = 'weight' if 'weight' in name else 'bias'
        #
        # table = wandb.Table(data=np.expand_dims(data.ravel(), axis=1).tolist(), columns=[column_name])
        #
        # experiment_logger.log({f'parameter_values_{stage}_training/{name}': wandb.plot.histogram(table=table,
        #                                                                                          value=column_name,
        #                                                                                          title=name)})


