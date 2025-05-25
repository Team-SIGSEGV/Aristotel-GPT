import math
import os

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling,
    pipeline
)

# 1. Конфигурация
BASE_MODEL = "DmitryYarov/Aristo2025"
DATASET_ID = "DmitryYarov/aristotle-russian"
OUTPUT_DIR = "./final_aristotle_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
DEBUG = True

# Проверка, обучена ли модель (существуют ли необходимые файлы)
model_exists = os.path.exists(os.path.join(OUTPUT_DIR, "pytorch_model.bin")) and \
               os.path.exists(os.path.join(OUTPUT_DIR, "tokenizer_config.json"))

# 2. Загрузка и разбивка датасета
dataset = load_dataset(DATASET_ID)
if "train" in dataset and "test" in dataset:
    raw_train = dataset["train"]
    raw_test = dataset["test"]
else:
    raw_splits = dataset["train"].train_test_split(test_size=0.1)
    raw_train = raw_splits["train"]
    raw_test = raw_splits["test"]

if DEBUG:
    raw_train = raw_train.select(range(2000))
    raw_test = raw_test.select(range(200))

# 3. Токенизация
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)


def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)


train_tok = raw_train.map(tokenize_fn, batched=True)
test_tok = raw_test.map(tokenize_fn, batched=True)

# 4. Загрузка или обучение модели
if model_exists:
    print("Загружаем уже обученную модель...")
    model = AutoModelForCausalLM.from_pretrained(OUTPUT_DIR).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR, use_fast=True)
else:
    print("Модель не найдена, запускаем обучение...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(DEVICE)

    # 5. Тренировка full fine‑tuning
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=1,
        weight_decay=0.01,
        learning_rate=5e-5,
        logging_steps=50,
        save_strategy="no",
        load_best_model_at_end=False,
        fp16=torch.cuda.is_available(),
        logging_dir="./logs",
        report_to="none"
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=test_tok,
        data_collator=data_collator
    )

    trainer.train()

    # 6. Сохранение
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Модель обучена")


# Функция collate для DataLoader — собирает батчи из списка примеров в тензоры с паддингом
def collate_fn(batch):
    input_ids = [example["input_ids"] for example in batch]
    attention_mask = [example["attention_mask"] for example in batch]
    batch_encoding = tokenizer.pad(
        {"input_ids": input_ids, "attention_mask": attention_mask},
        padding=True,
        return_tensors="pt"
    )
    return batch_encoding


# 7. Оценка perplexity на тестовом наборе
def compute_perplexity(eval_dataset):
    model.eval()
    dataloader = DataLoader(
        eval_dataset.remove_columns(["text"]),
        batch_size=8,
        shuffle=False,
        pin_memory=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    losses = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            losses.append(outputs.loss.item())
    return math.exp(sum(losses) / len(losses))


perplexity = compute_perplexity(test_tok)
print(f"Perplexity: {perplexity:.2f}")


# 8. CLI

def infer(prompt, max_length=200):
    gen = pipeline(
        "text-generation",
        model=OUTPUT_DIR,
        tokenizer=OUTPUT_DIR,
        device=0 if torch.cuda.is_available() else -1
    )
    out = gen(
        prompt,
        max_length=max_length,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        repetition_penalty=1.1,
        early_stopping=True
    )
    return out[0]["generated_text"]


if __name__ == "__main__":
    print(f"AristotleGPT ready on {DEVICE}. Type 'exit' to quit.")
    while True:
        q = input("Q: ")
        if q.lower() in ("exit", "quit"):
            break
        print("A:", infer(q))
