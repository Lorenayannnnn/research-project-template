"""
- Tokenizer classes / setup_tokenizer
- functions: tokenize, padding, collate_fn, setup_dataloader...
"""

from transformers import AutoTokenizer


def preprocess(configs, raw_datasets):
    """takes in the raw dataset and preprocesses it into the format we want"""

    tokenizer = create_tokenizer(configs)

    # shuffle the dataset
    raw_datasets = raw_datasets.shuffle(seed=configs.training_args.seed)
    tokenized_train_dataset, tokenized_eval_dataset, tokenized_predict_dataset = None, None, None

    if configs.training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if configs.data_args.max_train_samples is not None:
            configs.data_args.max_train_samples = min(configs.data_args.max_train_samples, len(train_dataset))
            train_dataset = train_dataset.select(range(configs.data_args.max_train_samples))
        tokenized_train_dataset = train_dataset.map(tokenize, fn_kwargs={"configs": configs, "tokenizer": tokenizer})
        # Print an example of the tokenized dataset
        decoded_text = tokenizer.decode(tokenized_train_dataset[0]['input_ids'])
        print("Example: ", decoded_text)

    if configs.training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation_matched" if configs.data_args.dataset_name == "mnli" else "validation"]
        if configs.data_args.max_eval_samples is not None:
            configs.data_args.max_eval_samples = min(configs.data_args.max_eval_samples, len(eval_dataset))
            eval_dataset = eval_dataset.select(range(configs.data_args.max_eval_samples))
        tokenized_eval_dataset = eval_dataset.map(tokenize, fn_kwargs={"configs": configs, "tokenizer": tokenizer})

    if configs.training_args.do_predict:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["validation_matched" if configs.data_args.dataset_name == "mnli" else "validation"]
        if configs.data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), configs.data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        tokenized_predict_dataset = predict_dataset.map(tokenize, fn_kwargs={"configs": configs, "tokenizer": tokenizer})

    return {
        "train": tokenized_train_dataset,
        "eval": tokenized_eval_dataset,
        "predict": tokenized_predict_dataset
    }

def create_tokenizer(configs):
    """creates the tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(
        configs.model_args.tokenizer_name if configs.model_args.tokenizer_name else configs.model_args.model_name_or_path,
        padding_side="left",
        add_bos_token=configs.data_args.add_bos_token,
        add_eos_token=configs.data_args.add_eos_token,
        cache_dir=configs.data_args.cache_dir,
        use_fast=configs.model_args.use_fast_tokenizer,
        revision=configs.model_args.model_revision,
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize(input_text, configs, tokenizer):
    result = tokenizer(
        input_text,
        truncation=True,
        max_length=configs.data_args.max_seq_length,
        pad_to_multiple_of=configs.data_args.pad_to_multiple_of
    )

    if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < configs.data_args.max_seq_length
            and configs.data_args.add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    # result["labels"] = result["input_ids"].copy()
    result["labels"] = [-100 if x == tokenizer.pad_token_id else x for x in result["input_ids"]]
    return result

