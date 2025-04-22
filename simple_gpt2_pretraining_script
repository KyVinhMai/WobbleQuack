import os
import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, DatasetDict

def preprocess_data(data_files, tokenizer, max_length=512):
    """
    Tokenize and preprocess datasets for training.
    """
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding=False, max_length=max_length)

    dataset = load_dataset("text", data_files=data_files)
    tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    return tokenized_dataset

def create_and_train_model(domain, data_files, model_save_path, vocab_size=50257, max_length=512, training_args=None):
    """
    Pretrain a GPT-2 model for a specific domain (DNA, Music, or Language).
    """
    print(f"Preparing to train model for {domain}...")

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    tokenized_data = preprocess_data(data_files, tokenizer, max_length)

    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=max_length,
        n_embd=768,
        n_layer=12,
        n_head=12
    )
    model = GPT2LMHeadModel(config)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    if training_args is None:
        training_args = TrainingArguments(
            output_dir=os.path.join(model_save_path, domain),
            overwrite_output_dir=True,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            learning_rate=5e-5,
            save_strategy="epoch",
            logging_dir=os.path.join(model_save_path, domain, "logs"),
            logging_steps=100
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["validation"] if "validation" in tokenized_data else None,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()

    model.save_pretrained(os.path.join(model_save_path, domain))
    tokenizer.save_pretrained(os.path.join(model_save_path, domain))

    print(f"Model for {domain} trained and saved at {model_save_path}/{domain}.")

data_paths = {
    "language": "path_to_language_dataset.txt"
}

save_path = "./trained_models"
os.makedirs(save_path, exist_ok=True)

for domain, data_file in data_paths.items():
    create_and_train_model(domain, data_file, save_path)
