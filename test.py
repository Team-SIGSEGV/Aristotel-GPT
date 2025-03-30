from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Загружаем токенизатор и модель
tokenizer = GPT2Tokenizer.from_pretrained("./aristotle_model")
model = GPT2LMHeadModel.from_pretrained("./aristotle_model")

def generate_text(prompt, max_length=200):
    # текст -> токены
    inputs = tokenizer(prompt, return_tensors="pt")

    # Переносим модель на устройство (GPU или CPU)
    model.eval()
    device = model.device

    outputs = model.generate(
        inputs.input_ids.to(device),
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_p=0.95,              # хз что эт
        temperature=0.7,         # крч если меньше значение, то более детерминированная генерация
    )

    # токены -> текст
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

prompt = "Человек должен стремиться к"
generated_text = generate_text(prompt)
print(generated_text)
