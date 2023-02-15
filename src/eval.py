import hydra
import pyrootutils
from omegaconf import DictConfig

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #


@hydra.main(version_base="1.3", config_path=str(root / "configs"), config_name="eval.yaml")
def main(cfg: DictConfig) -> None:

    from src.tasks.eval_task import evaluate

    evaluate(cfg)


if __name__ == "__main__":
    main()
