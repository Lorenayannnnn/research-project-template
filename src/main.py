import argparse

from src.common_utils import load_config_and_setup_output_dir, prepare_wandb
from src.data_module.load_data import load_data_from_hf, setup_dataloader
from src.data_module.preprocessing import preprocess
from src.model_module.load_model import load_model
from src.train_module.train_utils import create_trainer_args, create_trainer


def main(args):
    print("Loading configuration, setting up output directories...")
    configs = load_config_and_setup_output_dir(args)
    configs = prepare_wandb(configs)

    """Load the data"""
    raw_datasets = load_data_from_hf(configs.data_args.dataset_name, cache_dir=configs.data_args.cache_dir)

    """Preprocess data"""
    tokenized_datasets = preprocess(configs, raw_datasets)

    data_loader = setup_dataloader(datasets=tokenized_datasets, batch_size=configs.training_args.micro_batch_size)

    """Load model"""
    model = load_model(configs)

    """Set up trainer"""
    trainer_args = create_trainer_args(configs)
    # Setup trainer
    # trainer = create_trainer(model, data_loader["train"], data_loader["eval"], trainer_args)
    trainer = create_trainer(model, tokenized_datasets["train"], data_loader["eval"], trainer_args)

    if configs.training_args.do_train:
        print("Start training...")
        trainer.train()
        model.save_pretrained(configs.training_args.output_dir)
    elif configs.training_args.do_eval:
        print("Start evaluating...")
        # trainer.evaluate()
    elif configs.training_args.do_predict:
        print("Start predicting...")
        # trainer.predict()

    print("yay!")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_configs",
        type=str,
        required=True,
        help="Base configuration file"
    )

    parser.add_argument(
        "--overwrite_configs",
        type=str,
        required=False,
        help="Config/variation to overwrite the base configuration"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
