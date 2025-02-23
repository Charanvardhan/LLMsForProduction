"""
File: transformersArchitectures.py

Description:
This script demonstrates how to use three major transformer architectures—BART, BERT, and GPT-2—using the Hugging Face `transformers` library.
It covers three NLP tasks: summarization, text classification, and text generation, showcasing how each model architecture is suited for different use cases.

Sections:
1. **BART (Bidirectional and Auto-Regressive Transformer)** - Encoder-Decoder model for summarization.
2. **BERT (Bidirectional Encoder Representations from Transformers)** - Encoder-only model for text classification.
3. **GPT-2 (Generative Pre-trained Transformer 2)** - Decoder-only model for text generation.

Features:
- Demonstrates loading pre-trained models and their architectures.
- Shows how to use Hugging Face pipelines for quick implementation of NLP tasks.
- Provides examples with real outputs for better understanding.
"""

from transformers import AutoModel, AutoTokenizer
from transformers import pipeline

BART = AutoModel.from_pretrained("facebook/bart-large")

print(BART)

print("-"* 200)

# Create a summarization pipeline using the BART model fine-tuned on CNN/DailyMail
summerizer = pipeline("summarization", model="facebook/bart-large-cnn")
sum = summerizer("""Gaga was best known in the 2010s for pop hits like “Poker Face” and avant-garde experimentation on albums like “Artpop,” and Bennett, a singer who mostly stuck to standards, was in his 80s when the pair met. And
                  yet Bennett and Gaga became fast friends and close collaborators, which they remained until Bennett’s death at 96 on Friday. They recorded two albums together, 2014’s “Cheek to Cheek” and 2021’s “Love for Sale,
                 ” which both won Grammys for best traditional pop vocal album.""",min_length=20, max_length=50) #  controlled output length
print(sum[0]['summary_text'])

print("-"*200)

print("-"*100, "BERT", "-"*100)

BERT = AutoModel.from_pretrained("bert-base-uncased")

print(BERT)

print("-"*200)

# Create a text classification pipeline using a sentiment analysis model based on BERT.
# This model outputs sentiment labels like '1 star' to '5 stars'.
classifier = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")
# Classify the sentiment of the given text.
lbl = classifier("""This feels so great to learn new things""")

print(lbl)

print("-"*200)

print("-"*100, "GPT-2", "-"*100)

gpt2 = AutoModel.from_pretrained("gpt2")

print(gpt2)

generator = pipeline(model="gpt2")

# Generate multiple sentence completions given an initial prompt.
output = generator("this movie is very ", do_sample=True, top_p=0.95, num_return_sequences=4, max_new_tokens=50, return_full_text=False)

for item in output:
    print(">", item["generated_text"])

