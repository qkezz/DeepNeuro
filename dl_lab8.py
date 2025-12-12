from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

with open('ENG_article.txt', 'r', encoding='utf-8') as file:
    article_text = file.read()

system_prompt = """Ты — помощник, анализирующий текст. Прочитай предоставленный текст и на основе только его информации дай точные ответы на вопросы.
Если в тексте нет ответа, честно скажи об этом. Ответы давай на русском языке, цитируй фразы из текста для доказательства."""

user_questions = """
Ответь на следующие вопросы:

1. В каком году была обозначена проблема взрывающихся градиентов?
2. Кто в 1891 году разработал метод уничтожающей производной?
3. Кто предложил цепное правило дифференцирования и в каком году?
"""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": f"Текст статьи:\n{article_text}\n\n{user_questions}"}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(**model_inputs, max_new_tokens=512, temperature=0.1, do_sample=True)
generated_ids_ = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
response = tokenizer.batch_decode(generated_ids_, skip_special_tokens=True)[0]

with open('prompt.txt', 'w', encoding='utf-8') as f:
    f.write(text)

with open('res.txt', 'w', encoding='utf-8') as f:
    f.write(response)
