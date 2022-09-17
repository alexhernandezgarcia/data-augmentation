#!/bin/bash
# Run from root folder with: bash scripts/moons_run.sh

# run baseline experiments for various samples without any augmentation
python src/train.py -m experiment=moons_experiment.yaml trainer=gpu \
logger.wandb.project='moons_experiment_5' \
model.layers="[2,20,30,20,1]" \
model.lr=0.1 \
trainer.min_epochs=100 \
trainer.max_epochs=100 \
datamodule.train_dataset.n_samples=20,50,100,250,500,1000,5000,10000 \
datamodule.data_aug.n_augmentations=0

# run same experiment as above with less datapoints, but each original data point augmented [1,9] times
python src/train.py -m experiment=moons_experiment.yaml trainer=gpu \
logger.wandb.project='moons_experiment_5' \
model.layers="[2,20,30,20,1]" \
model.lr=0.1 \
trainer.min_epochs=100 \
trainer.max_epochs=100 \
datamodule.train_dataset.n_samples=5,10,20,50,100,250 \
datamodule.data_aug.n_augmentations="range(1, 10)"
