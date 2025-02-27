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

def prepare_folder(file_path):
    """Prepare a folder for a file"""
    folder = os.path.dirname(file_path)
    if not os.path.exists(folder):
        os.makedirs(folder)

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

def load_config_and_setup_output_dir(args):
    base_configs = args.base_configs
    overwrite_configs = args.overwrite_configs
    if not os.path.exists(args.base_configs):
        raise FileNotFoundError(f"Config file {args.base_config} does not exist")
    if args.overwrite_configs is not None and not os.path.exists(args.overwrite_configs):
        raise FileNotFoundError(f"Config file {args.overwrite_configs} does not exist")
    configs = OmegaConf.load(base_configs)

    output_dir = configs.training_args.output_dir

    if overwrite_configs is not None:
        overwrite_configs = OmegaConf.load(overwrite_configs)
        output_dir = overwrite_configs.training_args.output_dir

        # Merge base and overwrite configs
        configs = OmegaConf.merge(configs, overwrite_configs)

    if configs.training_args.resume_from_checkpoint is not None:
        # Load configs from checkpoint
        output_dir = os.path.dirname(configs.training_args.resume_from_checkpoint)
        loaded_configs = OmegaConf.load(os.path.join(output_dir, "configs.yaml"))
        loaded_configs.training_args.do_train = configs.training_args.do_train
        loaded_configs.training_args.do_predict = configs.training_args.do_predict
        loaded_configs.training_args.resume_from_checkpoint = configs.training_args.resume_from_checkpoint
        configs = loaded_configs

    # Prepare output directory
    configs.training_args.output_dir = output_dir
    prepare_folder(output_dir)
    OmegaConf.save(configs, os.path.join(output_dir, "configs.yaml"))

    # Save base and overwrite configs
    # OmegaConf.save(configs, os.path.join(configs.training_args.output_dir, "configs", "base_configs.yaml"))
    # OmegaConf.save(overwrite_configs, os.path.join(configs.training_args.output_dir, "configs", "overwrite_configs.yaml"))

    return configs
