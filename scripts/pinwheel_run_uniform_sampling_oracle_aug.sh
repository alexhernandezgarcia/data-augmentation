#!/bin/bash
# Run from root folder with: bash scripts/pinwheel_run_uniform_sampling_oracle_aug.sh

export EXPERIMENT_FILE="pinwheel_experiment_uniform_sampling_oracle_aug.yaml"
export PROJECT_NAME="pinwheel_experiment_uniform_sampling_oracle_aug_2"
export LEARNING_RATE=0.09

# run baseline experiments for various samples without any augmentation
python src/train.py \
-m experiment=$EXPERIMENT_FILE \
trainer=gpu +logger.wandb.entity=alex_data_augmentation \
logger.wandb.project=$PROJECT_NAME \
n_classes=3,4,5,6 \
model.layers=[2,20,30,20,'${n_classes}'] model.optimizer.lr=$LEARNING_RATE \
trainer.min_epochs=150 \
datamodule.train_dataset.n_samples=3,5,7,10,15,20,25,30,50,100 \
+logger.wandb.name='baseline_${datamodule.train_dataset.n_samples}_classes_${n_classes}'

# run same experiment as above with less datapoints, but each original data point augmented
python src/train.py \
-m experiment=$EXPERIMENT_FILE \
trainer=gpu +logger.wandb.entity=alex_data_augmentation \
logger.wandb.project=$PROJECT_NAME \
n_classes=3,4,5,6 \
model.layers=[2,20,30,20,'${n_classes}'] model.optimizer.lr=0.09 \
trainer.min_epochs=150 \
datamodule.train_dataset.n_samples=3,4,5,6,7 \
datamodule.data_aug.n_augmentations=4,6,9,12,15,50 \
datamodule.data_aug.max_d=0.1 \
datamodule.data_aug.lmd=0.01 \
val_test_noise=0.0 \
+logger.wandb.name='base_${datamodule.train_dataset.n_samples}_classes_${n_classes}_aug_${datamodule.data_aug.n_augmentations}_noise_${val_test_noise}'
