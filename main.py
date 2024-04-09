import numpy as np
import matplotlib.pyplot as plt
import pywt
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Set the model to evaluation mode and to use CPU
model.eval()
model.to("cpu")

# Generate token sequences on different topics
topics = ["The future of artificial intelligence", "Climate change and its impacts", "The role of social media in society"]
token_sequences = []

for topic in topics:
    input_ids = tokenizer.encode(topic, return_tensors="pt")  # Convert input text to tensor
    attention_mask = torch.ones(input_ids.shape)  # Create attention mask (all tokens are attended to)

    # Generate output using model
    output = model.generate(input_ids, attention_mask=attention_mask, max_length=100, num_return_sequences=1)
    token_sequence = tokenizer.decode(output[0], skip_special_tokens=True)
    token_sequences.append(token_sequence)

# Convert token sequences to numerical representations
numerical_sequences = []

for sequence in token_sequences:
    tokens = tokenizer.encode(sequence)
    numerical_sequences.append(tokens)

# Apply continuous wavelet transform and calculate power spectra
wavelet = 'mexh'  # Mexican hat wavelet
scales = np.arange(1, 32)

for i, sequence in enumerate(numerical_sequences):
    coef, freq = pywt.cwt(sequence, scales, wavelet)  # Use pywt.cwt here
    power = (abs(coef)) ** 2

    plt.figure(figsize=(12, 8))
    plt.imshow(power, cmap='coolwarm', aspect='auto')
    plt.xticks(np.arange(0, len(sequence), 10), np.arange(0, len(sequence), 10))
    plt.yticks(np.arange(0, len(scales), 4), scales[::4])
    plt.xlabel('Token Position')
    plt.ylabel('Scale')
    plt.title(f'Wavelet Power Spectrum - Topic {i+1}')
    plt.colorbar(label='Power')
    plt.tight_layout()
    plt.savefig(f'power_spectrum_topic_{i+1}.png')
    plt.close()

print("Wavelet power spectra generated and saved.")
print("Analyze the visualizations to identify patterns and structures at different scales.")
