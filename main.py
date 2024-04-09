import numpy as np
import matplotlib.pyplot as plt
import pywt
import torch
import os
import json
import datetime
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import cv2


# Load configurations
with open('config.json', 'r') as f:
    config = json.load(f)
topics = config['topics']

# Prepare environment
run_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
output_folder = Path(f"outputs/{run_id}")
output_folder.mkdir(parents=True, exist_ok=True)
log_file = output_folder / "log.txt"

# Load pre-trained GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model.eval()
model.to("cpu")

# Generate and process token sequences
token_sequences = []
numerical_sequences = []
for topic in topics:
    input_ids = tokenizer.encode(topic, return_tensors="pt")
    attention_mask = torch.ones(input_ids.shape)
    output = model.generate(input_ids, attention_mask=attention_mask, max_length=100, num_return_sequences=1)
    token_sequence = tokenizer.decode(output[0], skip_special_tokens=True)
    token_sequences.append(token_sequence)
    tokens = tokenizer.encode(token_sequence)
    numerical_sequences.append(tokens)

# Initialize logging
with open(log_file, 'w') as f:
    f.write(f"Run ID: {run_id}\nTopics: {topics}\nOutputs:\n")

# Analyze and visualize data
wavelet = 'morl'
scales = np.arange(1, 32)
images = []
for i, sequence in enumerate(numerical_sequences):
    coef, freq = pywt.cwt(sequence, scales, wavelet)
    power = (abs(coef)) ** 2

    plt.figure(figsize=(12, 8), dpi=300)
    plt.imshow(power, cmap='coolwarm', aspect='auto')
    plt.xticks(np.arange(0, len(sequence), 10), np.arange(0, len(sequence), 10))
    plt.yticks(np.arange(0, len(scales), 4), scales[::4])
    plt.xlabel('Token Position')
    plt.ylabel('Scale')
    plt.title(topics[i])
    plt.colorbar(label='Power')
    plt.tight_layout()

    img_path = output_folder / f"power_spectrum_topic_{i+1}.png"
    plt.savefig(img_path)
    plt.close()
    images.append(img_path)

    with open(log_file, 'a') as f:
        f.write(f"Generated {img_path}\n")

import cv2

# Create comparison images
for i in range(len(images) - 2):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), dpi=300)
    
    # Original images
    img1 = cv2.imread(str(images[i]))
    img2 = cv2.imread(str(images[i+1]))
    img3 = cv2.imread(str(images[i+2]))
    
    axs[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axs[0].set_title(topics[i])
    axs[0].axis('off')
    
    axs[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axs[1].set_title(topics[i+1])
    axs[1].axis('off')
    
    axs[2].imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
    axs[2].set_title(topics[i+2])
    axs[2].axis('off')
    
    # Difference images using image subtraction and thresholding
    diff1 = cv2.absdiff(img1, img2)
    diff1 = cv2.cvtColor(diff1, cv2.COLOR_BGR2GRAY)
    _, diff1 = cv2.threshold(diff1, 30, 255, cv2.THRESH_BINARY)
    
    diff2 = cv2.absdiff(img2, img3)
    diff2 = cv2.cvtColor(diff2, cv2.COLOR_BGR2GRAY)
    _, diff2 = cv2.threshold(diff2, 30, 255, cv2.THRESH_BINARY)
    
    fig2, axs2 = plt.subplots(1, 2, figsize=(12, 6), dpi=300)
    
    axs2[0].imshow(diff1, cmap='gray')
    axs2[0].set_title(f"Difference {topics[i]} - {topics[i+1]}")
    axs2[0].axis('off')
    
    axs2[1].imshow(diff2, cmap='gray')
    axs2[1].set_title(f"Difference {topics[i+1]} - {topics[i+2]}")
    axs2[1].axis('off')
    
    fig.tight_layout()
    fig2.tight_layout()
    
    comp_path = output_folder / f"comparison_{i+1}_{i+2}_{i+3}.png"
    diff_path = output_folder / f"difference_{i+1}_{i+2}_{i+3}.png"
    
    plt.figure(fig.number)
    plt.savefig(comp_path)
    plt.close()
    
    plt.figure(fig2.number)
    plt.savefig(diff_path)
    plt.close()
    
    with open(log_file, 'a') as f:
        f.write(f"Generated comparison: {comp_path}\n")
        f.write(f"Generated difference: {diff_path}\n")

print(f"All processing complete. Outputs and logs saved in {output_folder}.")