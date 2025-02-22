from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

model_path = "/Users/charanmannuru/Projects/LLMsForProduction/llama-3.2-local"

device = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype = torch.float16,
    device_map = {"" : device}
)

input = input()
tokens = tokenizer(input, return_tensors="pt").to(device)

output = model.generate(**tokens, max_length=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
