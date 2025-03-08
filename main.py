from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch

# 1. Загружаем датасет с Hugging Face
dataset = load_dataset("DmitryYarov/aristotle-russian")

# 2. Загружаем токенизатор и модель GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")

# 3. Функция токенизации
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# 4. Токенизируем датасет
tokenized_datasets = dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# 5. Параметры обучения
training_args = TrainingArguments(
    output_dir="./aristotle_model",
    eval_strategy="no",
    save_strategy="epoch",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    data_collator=data_collator
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"Using device: {device}")  # Должно вывести 'cuda' или 'cpu'

# 6. Запускаем обучение
trainer.train()

# 7. Сохраняем обученную модель
model.save_pretrained("./aristotle_model")
tokenizer.save_pretrained("./aristotle_model")