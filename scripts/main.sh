CONFIG_DIR=./configs
SRC_DIR=./src

export PYTHONPATH=:${PYTHONPATH}

#Check hydra.main decorator for config fn and use the bash script to override configs if needed
CUDA_VISIBLE_DEVICES=0 python ${SRC_DIR}/main.py

#bash scripts/main.sh