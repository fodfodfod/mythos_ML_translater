import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments
)
import warnings

# Suppress HF warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def main():
    # ===== CONFIG =====
    MODEL_NAME = "google/flan-t5-base"
    DATA_FILE = "data.json"
    OUTPUT_DIR = "../hugging_face/model"

    # ===== LOAD DATASET =====
    print("[DEBUG] Loading dataset...")
    dataset = load_dataset("json", data_files=DATA_FILE)["train"]
    print(f"[DEBUG] Dataset loaded: {len(dataset)} samples")

    # ===== DEVICE =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[DEBUG] Using device:", device)
    torch.backends.cudnn.benchmark = True

    # ===== LOAD MODEL + TOKENIZER =====
    print(f"[DEBUG] Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model = model.to(device)

    # ===== TOKENIZATION FUNCTION =====
    def preprocess(example):
        # T5 uses "input → label"
        model_input = tokenizer(
            example["input"],
            max_length=256,
            truncation=True
        )

        labels = tokenizer(
            example["output"],
            max_length=256,
            truncation=True
        )["input_ids"]

        model_input["labels"] = labels
        return model_input

    print("[DEBUG] Tokenizing...")
    tokenized_dataset = dataset.map(preprocess, batched=False)
    print("[DEBUG] Tokenization complete.")

    # ===== DATA COLLATOR =====
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # ===== TRAINING CONFIG =====
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=3e-4,
        num_train_epochs=10,
        bf16=True,
        fp16=False,
        save_strategy="epoch",
        logging_steps=10,
    )

    # ===== TRAINER =====
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # ===== START TRAINING =====
    print("[DEBUG] Starting training...")
    trainer.train()
    print("[DEBUG] Training complete.")

    # ===== SAVE MODEL =====
    print("[DEBUG] Saving final model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[DEBUG] Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
