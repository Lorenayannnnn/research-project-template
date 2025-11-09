import os
from random import random

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from transformers import set_seed


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)

def prepare_folder(output_dir):
    """Prepare a folder for a file"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def prepare_wandb(configs):
    # Check if parameter passed or if set within environ
    if configs.training_args.use_wandb and (len(configs.training_args.wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )) and configs.training_args.do_train:
        configs.training_args.use_wandb = True
        # Only overwrite environ if wandb param passed
        if len(configs.training_args.wandb_project) > 0:
            os.environ["WANDB_PROJECT"] = configs.training_args.wandb_project
        if len(configs.training_args.wandb_watch) > 0:
            os.environ["WANDB_WATCH"] = configs.training_args.wandb_watch
        if len(configs.training_args.wandb_log_model) > 0:
            os.environ["WANDB_LOG_MODEL"] = configs.training_args.wandb_log_model
        configs.wandb_run_name = configs.training_args.output_dir.split("/")[-1]
        wandb.init(project=os.environ["WANDB_PROJECT"], name=configs.wandb_run_name)
    else:
        configs.training_args.use_wandb = False
    return configs

def load_config_and_setup_output_dir(configs):
    output_dir = configs.training_args.output_dir

    if configs.training_args.resume_from_checkpoint is not None:
        # Load configs from checkpoint
        output_dir = os.path.dirname(configs.training_args.resume_from_checkpoint)
        loaded_configs = OmegaConf.load(os.path.join(output_dir, "configs.yaml"))
        loaded_configs.training_args.do_train = configs.training_args.do_train
        loaded_configs.training_args.do_predict = configs.training_args.do_predict
        loaded_configs.training_args.resume_from_checkpoint = configs.training_args.resume_from_checkpoint
        configs = loaded_configs

    # Prepare output directory: automatically craeted by hydra
    # configs.training_args.output_dir = output_dir
    # prepare_folder(output_dir)
    # OmegaConf.save(configs, os.path.join(output_dir, "configs.yaml"))

    return configs
