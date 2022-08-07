#!/bin/bash
# Run from root folder with: bash scripts/moons_run.sh

# run baseline experiments for various samples without any augmentation
python src/train.py -m experiment=moons_experiment.yaml logger.wandb.project='moons_experiment' \
datamodule.dataset.n_samples=20,50,100,250,500,1000,5000,10000 datamodule.data_aug.iterations=0

# run same experiment as above now with 5 augmentation n_augmentations
python src/train.py -m experiment=moons_experiment.yaml logger.wandb.project='moons_experiment' \
datamodule.dataset.n_samples=20,50,100,250,500,1000,5000,10000 datamodule.data_aug.iterations=5