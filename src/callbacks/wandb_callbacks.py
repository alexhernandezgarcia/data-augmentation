import subprocess
from pathlib import Path
from typing import List
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plotly.figure_factory as ff
import pytorch_lightning as pl
import seaborn as sn
import torch
import wandb
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score

from src.utils import pylogger

log = pylogger.get_pylogger(__name__)


def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """Safely get Weights&Biases logger from Trainer."""

    if trainer.fast_dev_run:
        raise Exception(
            "Cannot use wandb callbacks since pytorch lightning disables loggers in `fast_dev_run=true` mode."
        )

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    # if isinstance(trainer.logger, LoggerCollection):
    #     for logger in trainer.logger:
    #         if isinstance(logger, WandbLogger):
    #             return logger

    raise Exception(
        "You are using wandb related callback, but WandbLogger was not found for some reason..."
    )


class WatchModel(Callback):
    """Make wandb watch model at the beginning of the run."""

    def __init__(self, log: str = "gradients", log_freq: int = 100, log_graph=True):
        self.logging_type = log
        self.log_freq = log_freq
        self.log_graph = log_graph

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        logger.watch(
            model=trainer.model,
            log=self.logging_type,
            log_freq=self.log_freq,
            log_graph=self.log_graph,
        )


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

            # according to wandb docs this should also work, but it crashes
            # experiment.log(f{"confusion_matrix/{experiment.name}": plt})

            # close plot
            plt.close("all")

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

            # names should be unique or else charts from different experiments in wandb will overlap
            experiment.log({f"f1_p_r_heatmap/{experiment.name}": wandb.Image(plt)}, commit=False)

            # reset plot
            plt.close("all")

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
    """Logs decision boundary on the validation set.

    The decision boundary itself is saved as decision_boundary.png under the logs directory for the
    experiment
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

        # is the dataset multiclass?
        multiclass = True if pl_module.task == "multiclass" else False

        # get validation data
        val_data = trainer.datamodule.test_dataloader().dataset
        valX, valY = val_data.X, val_data.Y

        ckpt_path = trainer.checkpoint_callback.best_model_path  # best model
        log.info("Drawing decision boundary ...")
        log.info(f"Using best checkpoint: {ckpt_path}")
        model = pl_module.load_from_checkpoint(checkpoint_path=ckpt_path)

        pl_module.eval()  # put model in eval mode
        self._show_separation(
            model=model,
            experiment_logger=experiment,
            X=valX,
            y=valY,
            solid=False,
            multiclass=multiclass,
        )
        self._show_separation(
            model=model,
            experiment_logger=experiment,
            X=valX,
            y=valY,
            solid=True,
            multiclass=multiclass,
        )

    @staticmethod
    def _get_color2(label, prob):
        rows, cols = label.shape
        norm = plt.Normalize(0.0, 1.0, clip=False)
        flat_label = label.ravel()
        flat_prob = norm(prob.ravel())
        probcolormap = np.array([mpl.colormaps['Reds'], mpl.colormaps['Blues'],
                                 mpl.colormaps['Oranges'], mpl.colormaps['Greens'],
                                 mpl.colormaps['Purples'], mpl.colormaps['Greys']])
        output = np.empty((flat_label.shape[0], 4), dtype='uint8')

        for i in range(len(flat_label)):
            output[i] = list(probcolormap[flat_label[i]](flat_prob[i], bytes=True))

        return output.reshape((rows, cols, -1))

    def _show_separation(
        self,
        model: pl.LightningModule,
        experiment_logger: WandbLogger.experiment,
        X: np.ndarray,
        y: np.ndarray,
        save: bool = True,
        solid: bool = False,
        multiclass: bool = False
    ):
        """Plots and logs decision boundary for a model and dataset (X, y)

        Args:
            model (pl.LightningModule):  lightning module
            experiment_logger (WandbLogger.experiment):  Wandb experiment run logger
            X (np.ndarray): Dataset samples
            y (np.ndarray) : Dataset labels
            save (bool): whether to save decision boundary plot or not.
        """
        sn.set(style="darkgrid", font_scale=1.4)

        x_start, x_end = np.min(X[:, 0]) - 0.5, np.max(X[:, 0]) + 0.5
        y_start, y_end = np.min(X[:, 1]) - 0.5, np.max(X[:, 1]) + 0.5

        xx, yy = np.mgrid[x_start:x_end:0.01, y_start:y_end:0.01]
        grid = np.c_[xx.ravel(), yy.ravel()]
        batch = torch.from_numpy(grid).type(torch.float32).to(device=model.device)
        with torch.no_grad():
            if not multiclass:
                probs = torch.sigmoid(model(batch).reshape(xx.shape)).cpu()
                probs = probs.numpy().reshape(xx.shape)
            else:
                logits = model(batch)
                logits = logits.reshape((xx.shape[0], xx.shape[1], int(np.max(y)+1)))
                probs = torch.softmax(logits, dim=-1).cpu()
                # probs = torch.softmax(model(batch).reshape(xx.shape), dim=-1).cpu()
                probs = probs.numpy() #.reshape(xx.shape)

        solid_tag = ""
        if solid and not multiclass:
            probs = probs >= 0.5
            solid_tag = "_solid"

        elif solid and multiclass:
            # VERY HACKY!!!
            # add +1 so that all values are > 0, thus resultant color is dark!
            y_hat = np.argmax(probs, axis=-1)
            probs = y_hat + 1
            solid_tag = "_solid"

        elif not solid and multiclass:
            y_hat = np.argmax(probs, axis=-1)
            probs = np.max(probs, axis=-1)

        f, ax = plt.subplots(figsize=(16, 10))

        ax.set_title(f"Decision boundary {solid_tag}", fontsize=14)

        if not multiclass:
            cmap = "RdBu"
            contour = ax.contourf(xx, yy, probs, 25, cmap=cmap, vmin=0, vmax=1)
        else:
            # first get the colors
            pix_colors = self._get_color2(label=y_hat, prob=probs)
            # https: // stackoverflow.com / a / 49834186 / 4699994
            # very important!!! mgrid and meshgrid end up with different results need to transpose.
            ax.imshow(np.transpose(pix_colors, [1, 0, 2]), extent=(x_start, x_end, y_start, y_end), alpha=0.8, origin='lower')

        #     ax_c = f.colorbar(contour)
        #     ax_c.set_label("$P(y = 1)$")
        #     ax_c.set_ticks([0, .25, .5, .75, 1])

        if multiclass:
            # establish colors and colormap
            #  * color blind colors, from https://bit.ly/3qJ6LYL
            redish = '#d73027'
            orangeish = '#fc8d59'
            greenish = '#33ff33'
            blueish = '#4575b4'
            purpleish = '#b266ff'
            greyish = '#7F8C8D'
            colormap = np.array([redish, blueish, orangeish, greenish, purpleish, greyish])
            ax.scatter(
                X[:, 0],
                X[:, 1],
                c=colormap[(y.flatten().astype('int'))],
                s=50,
                edgecolor='white',
                linewidth=1,
            )

        else:
            ax.scatter(
                X[:, 0],
                X[:, 1],
                c=y,
                s=50,
                cmap=cmap,
                edgecolor="white",
                linewidth=1,
            )

        ax.set(xlabel="$X_1$", ylabel="$X_2$")
        if save:
            plt.savefig(self.dirpath + f"/decision_boundary{solid_tag}.png")

        experiment_logger.log(
            {f"decision_boundary/{experiment_logger.name}{solid_tag}": wandb.Image(plt)},
            commit=False,
        )
        # close plot
        plt.close("all")


class LogWeightBiasDistribution(Callback):
    """Logs weights and bias distribution.

    The distribution plots are saved user the distribution plots directory under the logs directory
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
            self.save_params_as_numpy(name, param.cpu().data.numpy(), stage="before")
            self.plot_distribution(
                name,
                param.cpu().data.numpy().ravel(),
                stage="before",
                experiment_logger=experiment,
            )

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

        ckpt_path = trainer.checkpoint_callback.best_model_path  # best model
        log.info("Drawing decision boundary ...")
        log.info(f"Using best checkpoint: {ckpt_path}")
        model = pl_module.load_from_checkpoint(checkpoint_path=ckpt_path)

        # get parameters and their names and send them to plot_distribution
        for name, param in model.named_parameters():
            # print(name, param)
            self.save_params_as_numpy(name, param.cpu().data.numpy(), stage="after")
            self.plot_distribution(
                name, param.cpu().data.numpy().ravel(), stage="after", experiment_logger=experiment
            )

    def save_params_as_numpy(self, param_name: str, data: np.ndarray, stage: str) -> None:
        # create path
        path = Path(self.dirpath)
        path = path / self._plots_directory / stage
        path.mkdir(parents=True, exist_ok=True)
        np.save(file=str(path / param_name), arr=data)  # save weights as a .npy file

    def plot_distribution_plotly(
        self, name: str, data: np.ndarray, stage: str, experiment_logger: WandbLogger.experiment
    ) -> None:
        if data.size <= 1:
            log.warning(
                f"Parameter {name} has just 1 element, can't plot it using Plotly, ignoring..."
            )
            return
        color = "firebrick" if "weight" in name else "cornflowerblue"
        fig = ff.create_distplot([data], [name], colors=[color], show_rug=False)
        # fig = ff.create_distplot([data], [name], colors=colors, show_rug=False)
        fig.update_layout(title_text=name)  # , title_x=0.5, title_font_size=20)
        experiment_logger.log({f"parameter_values_{stage}_training/{name}": fig}, commit=False)

    def plot_distribution(
        self, name: str, data: np.ndarray, stage: str, experiment_logger: WandbLogger.experiment
    ) -> None:
        """
        Plots the distribution of data using Seaborn Histplot plot, the plot is saved under the directory given by
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
        plt.rcParams.update({"font.size": 22})
        plt.title(name)
        # sn.kdeplot(data=data, shade=True, color='red' if "weight" in name else 'blue')
        sn.histplot(data=data, color="red" if "weight" in name else "blue", kde=True)
        plt.xlabel("Weight values")

        # create path
        path = Path(self.dirpath)
        path = path / self._plots_directory / stage
        path.mkdir(parents=True, exist_ok=True)
        plt.savefig(path / (name + ".png"))

        experiment_logger.log(
            {f"parameter_values_{stage}_training/{name}": wandb.Image(plt)}, commit=False
        )
        # close plots
        plt.close("all")

        # passing plt crashes the program bug report created.
        # https://github.com/wandb/wandb/issues/3987
        # experiment_logger.log({f'parameter_values_{stage}_training/{name}': plt})

        # column_name = 'weight' if 'weight' in name else 'bias'
        #
        # table = wandb.Table(data=np.expand_dims(data.ravel(), axis=1).tolist(), columns=[column_name])
        #
        # experiment_logger.log({f'parameter_values_{stage}_training/{name}': wandb.plot.histogram(table=table,
        #                                                                                          value=column_name,
        #                                                                                          title=name)})


class LogSklearnDatasetPlots(Callback):
    """Logs Sklearn dataset plots from the datamodule."""

    def __init__(self, dirpath: str):
        """
        Constructor for LogSklearnDatasetPlots callback
        Args:
            dirpath (str): path where to save the dataset plots
        """
        self.dirpath = dirpath

        # subdirectory where plots of before and after training can be found
        self._plots_directory = "dataset_plots"

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

        # is the dataset multiclass?
        multiclass = True if pl_module.task == "multiclass" else False

        # plot and log validation data
        dataset = trainer.datamodule.val_dataloader().dataset
        x_lim, y_lim = self.plot_dataset(
            dataset_name="Validation",
            data_X=dataset.X,
            data_Y=dataset.Y,
            experiment_logger=experiment,
            multiclass=multiclass,
        )

        # plot and log training data
        dataset = trainer.datamodule.train_dataloader().dataset
        self.plot_dataset(
            dataset_name="Train",
            data_X=dataset.X,
            data_Y=dataset.Y,
            experiment_logger=experiment,
            x_lim=x_lim,
            y_lim=y_lim,
            multiclass=multiclass,
        )

        # plot and log test data
        dataset = trainer.datamodule.test_dataloader().dataset
        self.plot_dataset(
            dataset_name="Test",
            data_X=dataset.X,
            data_Y=dataset.Y,
            experiment_logger=experiment,
            x_lim=x_lim,
            y_lim=y_lim,
            multiclass=multiclass,
        )

    def plot_dataset(
        self,
        dataset_name: str,
        data_X: np.ndarray,
        data_Y: np.ndarray,
        experiment_logger: WandbLogger.experiment,
        x_lim: tuple = None,
        y_lim: tuple = None,
        multiclass=False,
    ) -> tuple[tuple, tuple]:
        """
        Plots the scatter plot of a Sklearn dataset and logs to wandb as an image
        Args:
            dataset_name (str) : name of the dataset to plot, used in title
            data_X (np.ndarray) : Dataset samples
            data_Y (np.ndarray) : Dataset labels
            experiment_logger (WandbLogger.experiment): the wandb logger object
            x_lim (tuple): a left and right limit of the x-axis. Default None
            y_lim (tuple): a left and right limit of the y-axis. Default None

        Returns
            x_lim, y_lim (tuple[tuple, tuple]): a tuple with two tuples denoting the x_lim and y_lim of the plot that
                                                is created.
        """
        sn.set(style="darkgrid", font_scale=1.4)

        # set figure size
        plt.figure(figsize=(16, 10))
        # set font size
        plt.title(f"{dataset_name} Dataset ({len(data_X)} Samples)")

        if not multiclass:
            colors = np.array(["red", "green"])
            plt.scatter(data_X[:, 0], data_X[:, 1], color=list(colors[data_Y.flatten()]))
        else:
            # establish colors and colormap
            #  * color blind colors, from https://bit.ly/3qJ6LYL
            redish = '#d73027'
            orangeish = '#fc8d59'
            greenish = '#33ff33'
            blueish = '#4575b4'
            purpleish = '#b266ff'
            greyish = '#7F8C8D'
            colormap = np.array([redish, blueish, orangeish, greenish, purpleish, greyish])
            plt.scatter(data_X[:, 0], data_X[:, 1], color=colormap[(data_Y.flatten().astype('int'))])


        if x_lim and y_lim:
            plt.xlim(x_lim)
            plt.ylim(y_lim)

        # create path
        path = Path(self.dirpath)
        path = path / self._plots_directory
        path.mkdir(parents=True, exist_ok=True)
        plt.savefig(path / f"{dataset_name}_Dataset.png")

        experiment_logger.log({f"Charts/{dataset_name}_dataset": wandb.Image(plt)}, commit=False)
        x_lim, y_lim = plt.xlim(), plt.ylim()

        # close plots
        plt.close("all")

        return x_lim, y_lim


class AddToConfigEffectiveTrainSize(Callback):
    """Logs a simple metric: Effective Training Set Size (ETSS)

    ETSS = num_training_samples * (n_augmentations + 1 ) = Total training set size
    """

    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """
        Callback runs when training starts
        Args:
            trainer: lightning trainer
            pl_module: lightning module
        """

        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        # get train data
        train_data = trainer.datamodule.train_dataloader().dataset
        valX, valY = train_data.X, train_data.Y

        experiment.config.update({"effective_training_size": len(valY)})

        # hack to add base samples to the config
        if "datamodule/train_val_test_split" in logger.experiment.config._items:
            base_n_samples = logger.experiment.config._items["datamodule/train_val_test_split"][0]
            experiment.config.update({"datamodule/train_dataset/n_samples": base_n_samples})
