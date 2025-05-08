import torch
from transformers import AutoModel


def load_model(configs):
    """main function for loading the model_module"""
    model = AutoModel.from_pretrained(
        configs.training_args.resume_from_checkpoint if configs.training_args.resume_from_checkpoint else configs.model_args.model_name_or_path,
        cache_dir=configs.data_args.cache_dir,
        load_in_8bit=True,
        device_map="auto",
        torch_dtype=torch.float16
    )
    # Set up LoRA
    # if configs.training_args.resume_from_checkpoint:
    #     with open(os.path.join(configs.training_args.resume_from_checkpoint, "adapter_config.json")) as f:
    #         # Convert to dict
    #         config_params = json.load(f)
    #         config = LoraConfig(**config_params)
    #         if configs.training_args.do_train or (configs.training_args.do_predict and configs.training_args.save_grads):
    #             config.inference_mode = False
    # else:
    #     config = LoraConfig(
    #         r=configs.model_args.lora_r,
    #         lora_alpha=configs.model_args.lora_alpha,
    #         lora_dropout=configs.model_args.lora_dropout,
    #         target_modules=list(configs.model_args.lora_target_modules),
    #         bias="none",
    #         task_type="CAUSAL_LM"
    #     )
    # model = get_peft_model(model, config)
    # model.print_trainable_parameters()
    return model
