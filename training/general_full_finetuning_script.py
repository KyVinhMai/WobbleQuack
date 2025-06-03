import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union, Dict, List
from pathlib import Path # Import Path

import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)
from datasets import load_dataset, DatasetDict
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training


logger = logging.getLogger(__name__)

# IMPORTANT: Modify "username" and "project" as needed for your environment, or ensure this path is writable and appropriate.
FIXED_BASE_OUTPUT_DIR = Path("/pub/username/project")

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether or not to allow for custom models defined on the Hub in their own modeling files"},
    )
    hf_token: Optional[str] = field(
        default=None, metadata={"help": "Token to use for Hugging Face Hub authentication if needed."}
    )
    use_flash_attention_2: bool = field(
        default=False,
        metadata={"help": "Whether to use Flash Attention 2. Model must support it. Pass `attn_implementation='flash_attention_2'` to model loading."}
    )

@dataclass
class DataTrainingArguments:
    dataset_path: str = field(
        metadata={"help": "Path to the training dataset. Can be a directory, a single file (txt, jsonl, csv)."}
    )
    text_column: str = field(
        default="text",
        metadata={"help": "The name of the column in the datasets containing the text field."}
    )
    validation_dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the validation dataset. If None, will try to use validation_split_percentage."}
    )
    max_seq_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={"help": "The percentage of the train set used as validation set in case validation_dataset_path is not provided."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

@dataclass
class PeftArguments:
    use_lora: bool = field(default=False, metadata={"help": "Whether to use LoRA for parameter-efficient finetuning."})
    lora_r: int = field(default=8, metadata={"help": "LoRA attention dimension (rank)."})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha scaling factor."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout probability."})
    lora_target_modules: Optional[List[str]] = field(
        default=None, 
        metadata={"help": (
            "List of module names to apply LoRA to (e.g., 'q_proj', 'v_proj'). "
            "If None, PEFT library attempts to find suitable layers automatically. "
            "Common for Llama: ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], for GPT2: ['c_attn']"
        )}
    )
    lora_bias: str = field(
        default="none",
        metadata={"help": "Bias type for LoRA. Can be 'none', 'all' or 'lora_only'. Default is 'none'."}
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, PeftArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, peft_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, peft_args = parser.parse_args_into_dataclasses()

    # --- Construct Effective Output Directory ---
    # The user-provided `training_args.output_dir` will be treated as the run-specific subdirectory name.
    if not training_args.output_dir:
        logger.error("--output_dir must be specified (it will be used as the run-specific subdirectory name).")
        sys.exit(1)
    
    # Ensure output_dir is just a name, not a path that tries to escape the base.
    run_specific_subdir_name = Path(training_args.output_dir).name 
    effective_output_dir = FIXED_BASE_OUTPUT_DIR / run_specific_subdir_name
    training_args.output_dir = str(effective_output_dir) # Update TrainingArguments

    # --- WandB Setup ---
    if "wandb" not in training_args.report_to and os.environ.get("WANDB_PROJECT"):
        logger.info("WANDB_PROJECT environment variable set. Adding 'wandb' to report_to list.")
        if training_args.report_to is None or training_args.report_to == "all" or isinstance(training_args.report_to, str) and training_args.report_to == "none":
            training_args.report_to = ["wandb"]
        elif isinstance(training_args.report_to, list) and "wandb" not in training_args.report_to:
                training_args.report_to.append("wandb")
    
    if training_args.report_to and "wandb" in training_args.report_to:
        try:
            import wandb
            if not os.environ.get("WANDB_PROJECT"):
                logger.warning("Reporting to wandb, but WANDB_PROJECT environment variable is not set. Wandb might ask interactively or use a default project.")
            
            if training_args.run_name is None:
                # Create a default run name if not specified
                run_name_parts = [Path(model_args.model_name_or_path).name]
                if peft_args.use_lora:
                    run_name_parts.append("lora")
                run_name_parts.append(run_specific_subdir_name) # Use the run-specific name
                training_args.run_name = "-".join(run_name_parts)
            logger.info(f"WandB run name set to: {training_args.run_name}")

        except ImportError:
            logger.warning("wandb reporting is enabled, but `wandb` library is not installed. Please install it: `pip install wandb`")
            if isinstance(training_args.report_to, list):
                training_args.report_to = [r for r in training_args.report_to if r != "wandb"]
            elif training_args.report_to == "wandb":
                 training_args.report_to = "none"


    # --- Logging Setup ---
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Effective output directory: {training_args.output_dir}") # Log the final path
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}, bf16 training: {training_args.bf16}"
    )
    # Log all arguments
    logger.info(f"Training parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"PEFT parameters {peft_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # --- Load Tokenizer ---
    tokenizer_kwargs = {
        "token": model_args.hf_token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    tokenizer_name_or_path = model_args.tokenizer_name_or_path or model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            logger.info("Tokenizer pad_token_id not set. Using eos_token_id as pad_token_id.")
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.pad_token = tokenizer.eos_token
        else:
            logger.warning("Tokenizer does not have an eos_token_id. Adding a new pad token: '<|pad|>'.")
            tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

    # --- Load Model ---
    model_load_kwargs = {
        "token": model_args.hf_token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.use_flash_attention_2:
        if torch.__version__ >= "2.0.0":
            model_load_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Attempting to use Flash Attention 2.")
        else:
            logger.warning("Flash Attention 2 requested but PyTorch version is < 2.0.0. Ignoring.")

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_load_kwargs
    )

    if tokenizer.vocab_size > model.get_input_embeddings().weight.shape[0]:
        logger.info(f"Resizing model token embeddings from {model.get_input_embeddings().weight.shape[0]} to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    # --- PEFT (LoRA) Configuration ---
    if peft_args.use_lora:
        
        logger.info("Applying LoRA configuration...")
        if training_args.gradient_checkpointing:
            # For LoRA, if `gradient_checkpointing` is true in `TrainingArguments`,
            # PEFT's `prepare_model_for_kbit_training` or manually enabling it can be useful.
            # `prepare_model_for_kbit_training` is more for quantized models but also sets this up.
            # Let's enable it directly on the model if specified in training_args,
            # as `get_peft_model` itself doesn't automatically enable it based on TrainingArguments.
            model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing for the base model.")


        lora_config = LoraConfig(
            r=peft_args.lora_r,
            lora_alpha=peft_args.lora_alpha,
            target_modules=peft_args.lora_target_modules,
            lora_dropout=peft_args.lora_dropout,
            bias=peft_args.lora_bias,
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        logger.info("LoRA model created.")
        model.print_trainable_parameters()

    # --- Load and Preprocess Dataset (remains the same as previous version) ---
    logger.info(f"Loading dataset from {data_args.dataset_path}")
    file_extension = None
    actual_data_path = data_args.dataset_path
    if os.path.isfile(actual_data_path):
        file_extension = actual_data_path.split(".")[-1]

    raw_datasets = DatasetDict()
    load_kwargs = {"download_mode": "force_redownload" if data_args.overwrite_cache else "reuse_dataset_if_exists"}

    try:
        if file_extension in ["json", "jsonl"]:
            raw_datasets["train"] = load_dataset("json", data_files=actual_data_path, split="train", **load_kwargs)
        elif file_extension == "csv":
            raw_datasets["train"] = load_dataset("csv", data_files=actual_data_path, split="train", **load_kwargs)
        elif file_extension == "txt" or os.path.isdir(actual_data_path):
            data_files_arg = actual_data_path if file_extension=="txt" else None
            data_dir_arg = actual_data_path if os.path.isdir(actual_data_path) else None
            raw_datasets["train"] = load_dataset("text", data_files=data_files_arg, data_dir=data_dir_arg, split="train", **load_kwargs)
        else: 
            raw_datasets["train"] = load_dataset(actual_data_path, split="train", token=model_args.hf_token, **load_kwargs)
    except Exception as e:
        logger.error(f"Failed to load training dataset from {actual_data_path}. Error: {e}")
        sys.exit(1)

    if data_args.validation_dataset_path:
        logger.info(f"Loading validation dataset from {data_args.validation_dataset_path}")
        val_actual_data_path = data_args.validation_dataset_path
        val_file_extension = val_actual_data_path.split(".")[-1] if os.path.isfile(val_actual_data_path) else None
        try:
            if val_file_extension in ["json", "jsonl"]:
                 raw_datasets["validation"] = load_dataset("json", data_files=val_actual_data_path, split="train", **load_kwargs)
            elif val_file_extension == "csv":
                 raw_datasets["validation"] = load_dataset("csv", data_files=val_actual_data_path, split="train", **load_kwargs)
            elif val_file_extension == "txt" or os.path.isdir(val_actual_data_path):
                 val_data_files_arg = val_actual_data_path if val_file_extension=="txt" else None
                 val_data_dir_arg = val_actual_data_path if os.path.isdir(val_actual_data_path) else None
                 raw_datasets["validation"] = load_dataset("text", data_files=val_data_files_arg, data_dir=val_data_dir_arg, split="train", **load_kwargs)
            else: 
                raw_datasets["validation"] = load_dataset(val_actual_data_path, split="validation", token=model_args.hf_token, **load_kwargs)
        except Exception as e:
            logger.warning(f"Could not load validation dataset {val_actual_data_path}, trying 'train' split or skipping. Error: {e}")
    elif data_args.validation_split_percentage and data_args.validation_split_percentage > 0:
        if "train" in raw_datasets:
            logger.info(f"Splitting train dataset for validation ({data_args.validation_split_percentage}%).")
            split_dataset = raw_datasets["train"].train_test_split(test_size=data_args.validation_split_percentage / 100.0, shuffle=True, seed=training_args.seed)
            raw_datasets["train"] = split_dataset["train"]
            raw_datasets["validation"] = split_dataset["test"]
        else:
            logger.warning("Cannot create validation split as training data failed to load.")
    
    logger.info(f"Raw datasets loaded: {raw_datasets}")

    effective_max_length = data_args.max_seq_length
    if not effective_max_length:
        effective_max_length = tokenizer.model_max_length
        if not effective_max_length or effective_max_length > 2048: 
            effective_max_length = 1024 
            logger.info(f"max_seq_length not set, defaulting to {effective_max_length}.")
    
    if tokenizer.model_max_length and effective_max_length > tokenizer.model_max_length:
        logger.warning(
            f"Specified/defaulted max_seq_length ({effective_max_length}) is greater than the model's max length "
            f"({tokenizer.model_max_length}). Using model's max length: {tokenizer.model_max_length}."
        )
        effective_max_length = tokenizer.model_max_length

    def tokenize_function(examples):
        return tokenizer(examples[data_args.text_column], truncation=True, max_length=effective_max_length)

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=training_args.dataloader_num_workers,
            remove_columns=raw_datasets["train"].column_names if "train" in raw_datasets else None,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    
    if not tokenized_datasets.get("train") or not any(tokenized_datasets["train"]):
        logger.error("Training dataset is empty after tokenization. Check data, text_column, and tokenization.")
        sys.exit(1)

    # --- Initialize Trainer ---
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args, # training_args.output_dir is now the full path
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets.get("validation"),
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # --- Train ---
    logger.info("*** Starting Training ***")
    if training_args.resume_from_checkpoint:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        train_result = trainer.train()
    
    trainer.save_model() 
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    # --- Evaluate ---
    if training_args.do_eval and tokenized_datasets.get("validation"):
        logger.info("*** Starting Evaluation ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    logger.info(f"Finetuning script finished. Model and training artifacts saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()