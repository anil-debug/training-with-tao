# TAO Training Flow

## Introduction
This script automates the process of setting up the environment, downloading necessary tools, and training a DINO model using NVIDIA TAO (Transfer Learning Toolkit). 

## Prerequisites
- Docker and Docker Compose installed
- NVIDIA GPU with CUDA support (for training)
- Dataset prepared in COCO format and placed under `tao-experiments/data`

## Getting Started
1. Clone this repository.
2. Prepare your dataset according to the COCO format and place it under `tao-experiments/data`.
3. Run the following command to start the environment:
   ```bash
   docker-compose up
# Note

This script is written for training a DINO model with fan small as backbone.

You can modify the script to use different backbones or pretrained models available from NGC by adjusting the `pretrained_model` argument when creating the `TAOTrainingFlow` object.
