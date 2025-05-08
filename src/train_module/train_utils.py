import transformers

def create_trainer_args(configs):
    args = configs.training_args
    training_args = configs.training_args
    args = transformers.TrainingArguments(
        per_device_train_batch_size=training_args.micro_batch_size,
        gradient_accumulation_steps=training_args.batch_size // training_args.micro_batch_size,
        warmup_steps=100,
        num_train_epochs=training_args.num_train_epochs,
        learning_rate=training_args.learning_rate,
        fp16=True,
        logging_steps=10,
        optim="adamw_torch",
        evaluation_strategy="steps" if training_args.do_eval else "no",
        save_strategy=training_args.save_strategy,
        eval_steps=200 if training_args.do_eval else None,
        save_steps=200,
        output_dir=training_args.output_dir,
        save_total_limit=training_args.save_total_limit,
        load_best_model_at_end=True if training_args.do_eval else False,
        report_to="wandb" if configs.use_wandb else None,
        run_name=training_args.wandb_run_name if training_args.use_wandb else None,
    )
    return args


def create_trainer(model, train_data, val_data, trainer_args, data_collator=None):
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=trainer_args,
        data_collator=data_collator,
    )
    return trainer