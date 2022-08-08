#!/bin/bash
# Run from root folder with: bash scripts/moons_run.sh

# run baseline experiments for various samples without any augmentation
python src/train.py -m experiment=moons_experiment.yaml logger.wandb.project='moons_experiment' \
datamodule.train_dataset.n_samples=20,50,100,250,500,1000,5000,10000 datamodule.data_aug.n_augmentations=0

# run same experiment as above now with each original data point augmented 5 times
python src/train.py -m experiment=moons_experiment.yaml logger.wandb.project='moons_experiment' \
datamodule.train_dataset.n_samples=20,50,100,250,500,1000,5000,10000 datamodule.data_aug.n_augmentations=5