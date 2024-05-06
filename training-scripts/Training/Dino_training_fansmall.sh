#!/bin/bash

#!/bin/bash
export KEY='nvidia_tlt'
# Define the local project directory that needs to be mapped to the TAO docker session.
export LOCAL_PROJECT_DIR="/opt/nvidia/tools/tao-experiments"

# Set the HOST_DATA_DIR environment variable.
export HOST_DATA_DIR="${LOCAL_PROJECT_DIR}/data"

# Set the HOST_RESULTS_DIR environment variable.
export HOST_RESULTS_DIR="${LOCAL_PROJECT_DIR}/dino/results"

# Set the HOST_SPECS_DIR environment variable.
# Note: Make sure to set NOTEBOOK_ROOT appropriately before running the script.
export HOST_SPECS_DIR="$PWD/dino/specs"


# Define the path for the mounts file.
mounts_file="$HOME/.tao_mounts.json"
# Create the directories using mkdir -p.
mkdir -p "$HOST_DATA_DIR"
mkdir -p "$HOST_SPECS_DIR"
mkdir -p "$HOST_RESULTS_DIR"



# Create the TAO configurations in JSON format
tao_configs=$(cat <<EOF
{
    "Mounts": [
        {
            "source": "$LOCAL_PROJECT_DIR",
            "destination": "/opt/nvidia/tools/tao-experiments"
        },
        {
            "source": "$HOST_DATA_DIR",
            "destination": "/data"
        },
        {
            "source": "$HOST_SPECS_DIR",
            "destination": "/specs"
        },
        {
            "source": "$HOST_RESULTS_DIR",
            "destination": "/results"
        }
    ],
    "DockerOptions": {
        "shm_size": "16G",
        "ulimits": {
            "memlock": -1,
            "stack": 67108864
        },
        "user": "\$(id -u):\$(id -g)",
        "network": "host"
    }
}
EOF
)

# Write the mounts file.
echo "$tao_configs" > "$mounts_file"


# cat "$mounts_file"


# Verify the contents of the 'raw-data' directory
ls -l "$HOST_DATA_DIR/raw-data"


#!/bin/bash
# Set the environment variables
export CLI="ngccli_cat_linux.zip"
export LOCAL_PROJECT_DIR="/opt/nvidia/tools/tao-experiments"

# Create the ngccli directory
mkdir -p "$LOCAL_PROJECT_DIR/ngccli"

# Remove any previously existing CLI installations
rm -rf "$LOCAL_PROJECT_DIR/ngccli/*"

# Download the NGC CLI
wget "https://ngc.nvidia.com/downloads/$CLI" -P "$LOCAL_PROJECT_DIR/ngccli"

# Unzip the downloaded file
unzip -u "$LOCAL_PROJECT_DIR/ngccli/$CLI" -d "$LOCAL_PROJECT_DIR/ngccli/"

# Remove the downloaded zip file
rm "$LOCAL_PROJECT_DIR/ngccli/$CLI"

# Add ngc-cli to the PATH environment variable
export PATH="$LOCAL_PROJECT_DIR/ngccli/ngc-cli:$PATH"

# Run the NGC CLI command to list the models
ngc registry model list nvidia/tao/pretrained_dino_nvimagenet:*

# Pull pretrained model from NGC
ngc registry model download-version nvidia/tao/pretrained_dino_nvimagenet:fan_base_hybrid_nvimagenet --dest "$LOCAL_PROJECT_DIR/dino/"

# Check that model is downloaded into dir
echo "Check that model is downloaded into dir."
ls -l "$LOCAL_PROJECT_DIR/dino/pretrained_dino_nvimagenet_vfan_base_hybrid_nvimagenet/" 

# Print the contents of the train.yaml file
cat "$HOST_SPECS_DIR/train.yaml"

# Setting environment variables for data, specs, and results directories
DATA_DIR="/data"
SPECS_DIR="/specs"
RESULTS_DIR="/results"

export DATA_DIR
export SPECS_DIR
export RESULTS_DIR
# The environment variable HOST_DATA_DIR will be echoed
echo $HOST_DATA_DIR

# Print the messages
echo "For multi-GPU, change num_gpus in train.yaml based on your machine or pass --gpus to the cli."
echo "For multi-node, change num_gpus and num_nodes in train.yaml based on your machine or pass --num_nodes to the cli."

# If you face out of memory issue, you may reduce the batch size in the spec file by passing dataset.batch_size=2

# Run the tao command for dino training
# tao model dino train -e $SPECS_DIR/train.yaml results_dir=$RESULTS_DIR/

# dino train -e $SPECS_DIR/train.yaml -r $RESULTS_DIR -k $KEY --gpus 2
dino train -e $HOST_SPECS_DIR/train.yaml -r $HOST_RESULTS_DIR 

#!/bin/bash
echo 'Trained checkpoints:'
echo '---------------------'
ls -ltrh $HOST_RESULTS_DIR/train

#!/bin/bash

# Set the NUM_EPOCH environment variable
#!/bin/bash

# Set the NUM_EPOCH environment variable
NUM_EPOCH=098
export NUM_EPOCH

# Get the name of the checkpoint corresponding to the set epoch
CHECKPOINT=$(ls $HOST_RESULTS_DIR/train/*.pth | grep "epoch=$NUM_EPOCH" | head -n 1)
export CHECKPOINT

# Display the message
echo "Rename a trained model: "
echo "---------------------"

# Copy the checkpoint to a new location
cp $CHECKPOINT $HOST_RESULTS_DIR/train/dino_model.pth

# List the new model file
ls -ltrh $HOST_RESULTS_DIR/train/dino_model.pth

#!/bin/bash

# Evaluate on TAO model
dino evaluate \
    -e $HOST_SPECS_DIR/evaluate.yaml \
    evaluate.checkpoint=$HOST_RESULTS_DIR/train/dino_model.pth \
    results_dir=$HOST_RESULTS_DIR/

# Copy classmap to annotation directory
p $HOST_SPECS_DIR/classmap.txt $HOST_DATA_DIR/raw-data/annotations/

# Run TAO model inference for DINO
dino inference \
    -e $HOST_SPECS_DIR/infer.yaml \
    inference.checkpoint=$HOST_RESULTS_DIR/train/dino_model.pth \
    results_dir=$HOST_RESULTS_DIR/


# Export the RGB model to ONNX model
# dino export \
#     -e $HOST_SPECS_DIR/export.yaml \
#     export.checkpoint=$HOST_RESULTS_DIR/train/dino_model.pth \
#     export.onnx_file=$HOST_RESULTS_DIR/export/dino_model.onnx \
#     results_dir=$HOST_RESULTS_DIR/

# Generate TensorRT engine using tao deploy
# dino gen_trt_engine \
#     -e $HOST_SPECS_DIR/gen_trt_engine.yaml \
#     gen_trt_engine.onnx_file=$HOST_RESULTS_DIR/export/dino_model.onnx \
#     gen_trt_engine.trt_engine=$HOST_RESULTS_DIR/gen_trt_engine/dino_model.engine \
#     results_dir=$HOST_RESULTS_DIR

# Evaluate with generated TensorRT engine
# dino evaluate \
#     -e $HOST_SPECS_DIR/evaluate.yaml \
#     evaluate.trt_engine=$HOST_RESULTS_DIR/gen_trt_engine/dino_model.engine \
#     results_dir=$HOST_RESULTS_DIR/

# dino inference -e $HOST_SPECS_DIR/infer.yaml \
#     inference.trt_engine=$HOST_RESULTS_DIR/gen_trt_engine/dino_model.engine \
#     results_dir=$HOST_RESULTS_DIR/