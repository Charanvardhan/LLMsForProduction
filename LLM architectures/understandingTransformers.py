"""
File: architectureInAction.py

Description:
This script demonstrates the architecture and inner workings of the OPT-1.3B causal language model using the Hugging Face Transformers library. 
It walks through the process of tokenizing input text, generating token embeddings, adding positional embeddings, and applying the first layer 
of self-attention in the decoder.

Steps Covered:
1. Load the OPT-1.3B model and tokenizer.
2. Tokenize a sample input sentence into input IDs and attention masks.
3. Convert tokens into embeddings using the model's embedding layer.
4. Apply learned positional embeddings to incorporate token position information.
5. Sum token and positional embeddings to form the input to the transformer layers.
6. Pass the embeddings through the first self-attention layer to observe how the model attends to different tokens.

Outputs:
- Tokenized input and input IDs.
- Token embeddings and their shape.
- Positional embeddings and their shape.
- Result of the self-attention layer including updated hidden states.

Purpose:
This script helps visualize and understand how each component of the transformer architecture processes input text, focusing on embeddings 
and the self-attention mechanism in a causal language model setup.

Requirements:
- transformers (Hugging Face library)
- torch (PyTorch library)
"""


from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

load_bit_Config = BitsAndBytesConfig(load_in_8bit=True)
OPT = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b")

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")

input = "The quick brown fox jumps over the lazy dog"
inp_tokenized = tokenizer(input, return_tensors="pt")
print(inp_tokenized["input_ids"].size())
print(inp_tokenized)

print("-"*200)

print(OPT.model)

print("-"*200)

embedded_input = OPT.model.decoder.embed_tokens(inp_tokenized['input_ids'])
print("Layers: \t", OPT.model.decoder.embed_tokens)
print("Size: \t", embedded_input.size())
print("Output: \t", embedded_input)

print("-"*200)

embed_pos_input = OPT.model.decoder.embed_positions(inp_tokenized['attention_mask'])
print("Layer: \t", OPT.model.decoder.embed_positions)
print("Size: \t", embed_pos_input.size())
print("Output: \t", embed_pos_input)

print("-"*200)

embed_position_input = embedded_input + embed_pos_input
hidden_states, _, _ = OPT.model.decoder.layers[0].self_attn(embed_position_input)
print("Layer: \t", OPT.model.decoder.layers[0].self_attn)
print("Size: \t", hidden_states.size())
print("output: \t", hidden_states)