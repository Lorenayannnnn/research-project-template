
import hydra

from src.common_utils import load_config_and_setup_output_dir, prepare_wandb
from src.data_module.load_data import load_data_from_hf, setup_dataloader
from src.data_module.preprocessing import preprocess
from src.model_module.load_model import load_model
from src.train_module.train_utils import create_trainer_args, create_trainer

@hydra.main(config_path="configs", config_name="base_configs", version_base=None)
def main(configs):
    print("Loading configuration, setting up output directories...")
    configs = load_config_and_setup_output_dir(configs)
    configs = prepare_wandb(configs)

    """Load the data"""
    raw_datasets = load_data_from_hf(configs.data_args.dataset_name, cache_dir=configs.data_args.cache_dir)

    """Preprocess data"""
    tokenized_datasets, tokenizer = preprocess(configs, raw_datasets)
    # data_loaders = setup_dataloader(input_datasets=tokenized_datasets, batch_size=configs.running_args.micro_batch_size, tokenizer=tokenizer)

    """Load model"""
    model = load_model(configs)

    """Set up trainer"""
    trainer_args = create_trainer_args(configs)
    # Setup trainer
    trainer = create_trainer(model, tokenized_datasets["train"], tokenized_datasets["eval"], trainer_args)

    if configs.training_args.do_train:
        print("Start training...")
        trainer.train()
        model.save_pretrained(configs.training_args.output_dir)
    elif configs.training_args.do_eval:
        print("Start evaluating...")
        trainer.evaluate()
    elif configs.training_args.do_predict:
        print("Start predicting...")
        trainer.predict()

    print("yay!")


if __name__ == "__main__":
    main()
