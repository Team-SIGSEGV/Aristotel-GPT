import random

import optuna
import torch
from datasets import load_dataset
from evaluate import load
from pymorphy2 import MorphAnalyzer
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, \
    EarlyStoppingCallback


def load_data():
    dataset = load_dataset("DmitryYarov/aristotle-russian", split="train")
    dataset = dataset.train_test_split(test_size=0.1)
    return dataset


class TextAugmenter:
    def __init__(self):
        self.morph = MorphAnalyzer()
        self.tokenizer = GPT2Tokenizer.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")

    def augment_text(self, text, noise_prob=0.1, mask_prob=0.1):
        """
        Улучшенная аугментация текста:
        1. Лемматизация через pymorphy2
        2. Добавление шума (замена/удаление слов)
        3. Случайное маскирование токенов
        """
        tokens = self.tokenizer.tokenize(text)

        # 1001ая попытка аугментации
        processed_tokens = []
        for token in tokens:
            if token in self.tokenizer.all_special_tokens:
                processed_tokens.append(token)
                continue

            if random.random() < noise_prob:
                action = random.choice(["delete", "replace", "mask"])

                if action == "delete":
                    continue
                elif action == "replace":
                    new_token = random.choice(list(self.tokenizer.get_vocab().keys()))
                    processed_tokens.append(new_token)
                elif action == "mask" and random.random() < mask_prob:
                    processed_tokens.append("[MASK]")
                else:
                    processed_tokens.append(token)
            else:
                try:
                    lemma = self.morph.parse(token)[0].normal_form
                    processed_tokens.append(lemma)
                except:
                    processed_tokens.append(token)

        return self.tokenizer.convert_tokens_to_string(processed_tokens)


def tokenize_data(dataset, tokenizer):
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized = dataset.map(tokenize_fn, batched=True)
    return tokenized


def optimize_hyperparams(trial, model, tokenized_datasets):
    # решил сделать перебор параметров
    args = TrainingArguments(
        output_dir="./optuna_trials",
        learning_rate=trial.suggest_float("lr", 1e-5, 5e-4, log=True),
        per_device_train_batch_size=trial.suggest_categorical("batch_size", [8, 16]),
        num_train_epochs=3,
        weight_decay=trial.suggest_float("weight_decay", 0.0, 0.1),
        evaluation_strategy="epoch",
        logging_steps=50,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    trainer.train()
    eval_result = trainer.evaluate()
    return eval_result["eval_loss"]


def train_model(model, tokenizer, dataset, best_params):
    training_args = TrainingArguments(
        output_dir="./best_model",
        learning_rate=best_params["lr"],
        per_device_train_batch_size=best_params["batch_size"],
        num_train_epochs=5,
        weight_decay=best_params["weight_decay"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=100,
        gradient_accumulation_steps=2,
        fp16=torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()
    return trainer


def evaluate_model(model, tokenizer, dataset):
    perplexity = load("perplexity")
    results = perplexity.compute(
        model=model,
        add_start_token=True,
        texts=dataset["test"]["text"][:100]
    )
    print(f"Perplexity: {results['mean_perplexity']:.2f}")


def generate_text(model, tokenizer, prompt):
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    output = generator(
        prompt,
        max_length=100,
        num_beams=5,
        temperature=0.7,
        early_stopping=True
    )
    print("Generated text:", output[0]["generated_text"])


if __name__ == "__main__":
    dataset = load_data()

    augmenter = TextAugmenter()
    dataset = dataset.map(lambda x: {"text": augmenter.augment_text(x["text"])})

    tokenizer = GPT2Tokenizer.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")

    tokenized_datasets = tokenize_data(dataset, tokenizer)

    study = optuna.create_study(direction="minimize", pruner=MedianPruner())
    study.optimize(
        lambda trial: optimize_hyperparams(trial, model, tokenized_datasets),
        n_trials=5,
        timeout=3600
    )
    print("Best hyperparams:", study.best_params)

    trainer = train_model(model, tokenizer, tokenized_datasets, study.best_params)

    model.save_pretrained("./final_aristotle_model")
    tokenizer.save_pretrained("./final_aristotle_model")

    evaluate_model(model, tokenizer, dataset)

    generate_text(model, tokenizer, "Аристотель считал, что")
