from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)



# # Choose the model (Example: LLaMA 3 - 8B)
# model_name = "meta-llama/Meta-Llama-3-8B"

# # Load tokenizer & model
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# # Create a pipeline for text generation
# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# # Example query
# response = pipe("Translate 'Hello, how are you?' to French.", max_length=100)
# print(response[0]['generated_text'])
