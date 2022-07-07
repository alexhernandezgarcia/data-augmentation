import yaml
from typing import Any, Optional, List
from pydantic.dataclasses import dataclass


@dataclass
class Model:
    layers: List[int]


@dataclass
class Dataset:
    base_num_samples: List[int]
    noise: float = None
    shuffle: bool = True
    val_split: float = 0.2
    batch_size: int = 32


@dataclass
class Augmentation:
    type: str
    noise: float = 0.05
    num_of_iterations: int = 1


@dataclass
class Train:
    loss_func: str
    optimizer: str
    lr: float = 0.5  # learning rate
    epochs: int = 30


@dataclass(frozen=True)
class Config:
    model: Model
    dataset: Dataset
    train: Train
    augmentation: Optional[Augmentation] = None


def get_config_from_yaml(yaml_file) -> Config:
    """
    Get the config from yaml file
    Input:
        - yaml_file: yaml configuration file
    Return:
        - config: namespace
        - config_dict: dictionary
    """

    with open(yaml_file) as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)

    # convert the dictionary to a namespace using bunch lib
    config = Config(**config)
    return config
