"""
Movie Emoji Generator using LLaMA 3.2

This script loads a locally stored LLaMA 3.2 model and generates relevant 
emojis based on a given movie name. It uses a few-shot learning approach, 
where a predefined prompt includes example movie names paired with meaningful 
emojis. The model then generates emojis for a user-provided movie.

Key Features:
- Utilizes a locally stored LLaMA model for text generation.
- Implements few-shot learning to provide structured emoji responses.
- Optimized for Apple MPS (Metal Performance Shaders) for fast inference on Mac GPUs.
- Limits output to emojis, avoiding unnecessary text generation.

Usage:
1. Run the script.
2. Enter a movie name when prompted.
3. The model will generate a set of emojis representing the movie's themes, 
   characters, or events.
"""



from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

model_path = "/Users/charanmannuru/Projects/LLMsForProduction/llama-3.2-local"

device = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(model_path,
                                             torch_dtype=torch.float16,
                                             device_map={"": device})

few_shot_prompt = """
I will give you a movie name, and you will provide a set of emojis that represent the movie's key themes, characters, or events.

Examples:
🎥 **Titanic** - 🚢❄️💑💔🆘  
🎥 **The Matrix** - 🕶️💊💻🤖🔫  
🎥 **Inception** - 🌀🛌💤⏳🔍  
🎥 **Interstellar** - 🚀🌌⏳🌍👨‍👧  
🎥 **The Dark Knight** - 🦇🎭🃏🔥🏙️  
🎥 **Avengers: Endgame** - 🦸‍♂️🦸‍♀️💎⚡🛡️  
🎥 **Harry Potter** - ⚡📖🧙‍♂️🏰🐍  
🎥 **The Lion King** - 🦁👑🌅🎶💔  
🎥 **RRR** - 🔥🤝⚔️🇮🇳🐅  

Now, provide emojis for:
🎥 **
"""


movie_name = input("enter movie name :")
full_prompt = few_shot_prompt + movie_name + "** -"

tokens = tokenizer(full_prompt, return_tensors="pt").to(device)

output = model.generate(**tokens, max_new_tokens=10)
print(tokenizer.decode(output[0], skip_special_tokens=True))