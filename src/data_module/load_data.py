"""
- classes: Dataset, dataloader...
- functions: load_data_to_pd, collate_fn, setup_dataloader
"""

import json
from torch.utils.data import DataLoader
import pandas as pd

import datasets
from datasets import load_dataset, Dataset


def load_data_from_hf(file_or_dataset_name, cache_dir=None):
    """processes data into huggingface dataset"""
    if not file_or_dataset_name.endswith(".csv") and not (file_or_dataset_name.endswith(".json") or file_or_dataset_name.endswith(".jsonl")):
        raw_datasets = load_dataset(file_or_dataset_name, cache_dir=cache_dir)
    elif file_or_dataset_name.endswith(".csv"):
        # Loading a dataset from local csv files
        raw_datasets = load_dataset(
            "csv",
            data_files=file_or_dataset_name,
            cache_dir=cache_dir,
        )
    elif file_or_dataset_name.endswith(".json") or file_or_dataset_name.endswith(".jsonl"):
        # Loading a dataset from local json files
        raw_datasets = load_dataset(
            "json",
            data_files=file_or_dataset_name,
            cache_dir=cache_dir,
        )
    else:
        raise ValueError(f"unknown dataset format {file_or_dataset_name}")
    return raw_datasets


def load_data_to_pd(file_or_dataset_name, return_df_only=False):
    if file_or_dataset_name.endswith("jsonl"):
        with open(file_or_dataset_name, "r") as f:
            lines = f.readlines()
            data = [json.loads(line) for line in lines]
            if return_df_only:
                return pd.DataFrame(data)
            return datasets.Dataset.from_pandas(pd.DataFrame(data))
    pass


def collate_fn(batch):
    """
    collate function for DataLoader
    :param batch: list of samples
    :return: batch
    """
    pass


def setup_dataloader(datasets, batch_size):
    """

    :param datasets: dictionary of datasets (train, eval, predict)
    :param is_train: boolean
    :param batch_size: encoded test dataset
    :return:
    """
    dataloaders = {}
    for split in ["train", "eval", "predict"]:
        if split not in datasets:
            datasets[split] = None
        else:
            dataloaders[split] = DataLoader(datasets[split], shuffle=split == "train", batch_size=batch_size, collate_fn=collate_fn)
    return dataloaders