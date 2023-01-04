#!/bin/bash
# Run from root folder with: bash scripts/moons_run.sh

export EXPERIMENT_FILE="moons_experiment.yaml"
export PROJECT_NAME="moons_experiment_6_bug_fixed"
export LEARNING_RATE=0.5

# run baseline experiments for various samples without any augmentation
python src/train.py -m experiment=$EXPERIMENT_FILE trainer=gpu \
+logger.wandb.entity=alex_data_augmentation \
logger.wandb.project=$PROJECT_NAME \
model.layers="[2,20,30,20,1]" \
model.lr=$LEARNING_RATE \
trainer.min_epochs=100 \
trainer.max_epochs=100 \
datamodule.train_dataset.n_samples=6,10,15,20,25,30,35,40,45,50,100 \
datamodule.data_aug.n_augmentations=0

# run same experiment as above with less datapoints, but each original data point augmented [1,9] times
python src/train.py -m experiment=$EXPERIMENT_FILE trainer=gpu \
+logger.wandb.entity=alex_data_augmentation \
logger.wandb.project=$PROJECT_NAME \
model.layers="[2,20,30,20,1]" \
model.lr=$LEARNING_RATE \
trainer.min_epochs=100 \
trainer.max_epochs=100 \
datamodule.train_dataset.n_samples=6,8,10,12,16 \
datamodule.data_aug.n_augmentations=4,6,9,12,15
