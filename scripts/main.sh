CONFIG_DIR=./configs
SRC_DIR=./src

#This exits the script if any command fails
set -e

export PYTHONPATH=:${PYTHONPATH}

### START EDITING HERE ###
base_config="base_configs.yaml"  # This is the base config file
overwrite_config="overwrite_configs.yaml"  # Config for specific experiment

CUDA_VISIBLE_DEVICES=0 python ${SRC_DIR}/main.py \
    --base_configs=${CONFIG_DIR}/${base_config} \
    --overwrite_configs=${CONFIG_DIR}/${overwrite_config} \
