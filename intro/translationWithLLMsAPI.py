
from dotenv import load_dotenv
import os
import openai

import requests

# Load environment variables from .env file
load_dotenv("/Users/charanmannuru/Projects/LLMsForProduction/intro/openai.env")

API_KEY = os.getenv("OPEN_AI_KEY")

API_URL ="https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b"

HEADERS = {"Authorization": f"Bearer {API_KEY}"}

def query(payload):
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    return response.json()

response = query({"inputs": "Translate 'Hello, how are you?' to French."})
print(response)



# load_dotenv('/Users/charanmannuru/Projects/LLMsForProduction/intro/openai.env')

# # Ensure API key is loaded
# openai.api_key = os.getenv("OPEN_AI_KEY")

# input = "Hello, how are you?"

# respone = openai.ChatCompletion.create(
#     model="omni-moderation-latest",
#   messages=[
#     {"role": "system", "content": "you are a helpful assiatant"},
#         {"role": "user", "content": f"translate the following into french: {input}"}

#   ]
# )

# print(respone['choices'][0]['message']['context'])