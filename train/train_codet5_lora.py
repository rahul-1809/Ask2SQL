"""LoRA fine-tuning script for CodeT5+ on b-mc2/sql-create-context.
Run on Kaggle or local GPU. Minimal example.
"""
import os
from dataclasses import dataclass
from typing import Dict

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Trainer, TrainingArguments

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    raise ImportError("Please install peft: pip install peft")

MODEL_NAME = os.getenv("BASE_MODEL", "Salesforce/codet5-small")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./models/fine_tuned_text2sql_codet5")


@dataclass
class Sample:
    prompt: str
    target: str


def build_prompt(context: str, question: str) -> str:
    return f"Schema: {context}\nQuestion: {question}\nSQL:"


def preprocess(example: Dict) -> Dict:
    prompt = build_prompt(example["context"], example["question"])
    target = example["answer"].strip()
    return {"prompt": prompt, "target": target}


def main():
    dataset = load_dataset("b-mc2/sql-create-context")
    dataset = dataset.map(preprocess)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        inputs = tokenizer(batch["prompt"], truncation=True, max_length=512)
        labels = tokenizer(batch["target"], truncation=True, max_length=256)
        inputs["labels"] = labels["input_ids"]
        return inputs

    tokenized = dataset.map(tokenize, batched=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q", "v"],  # heuristic; adjust for CodeT5 architecture
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    args = TrainingArguments(
        OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-4,
        weight_decay=0.01,
        logging_steps=50,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=500,
        eval_steps=500,
        bf16=torch.cuda.is_available(),
        gradient_accumulation_steps=1,
        predict_with_generate=True,
        report_to=["none"],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation") or tokenized["train"].select(range(100)),
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Fine-tuning complete. Model saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
