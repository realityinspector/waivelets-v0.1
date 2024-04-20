import datetime
import json
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from matplotlib import transforms
from sentence_transformers import SentenceTransformer



# Load configurations
#with open('config.json', 'r') as f:
#    config = json.load(f)
#topics = config['topics']

# multi sentence content
with open('content.json', 'r') as f:
  content = json.load(f)
yeats = content['texts']['yeats']
oreilly = content['texts']['oreilly']

# cluster data processed offline
file = open("yeatsRes.csv", "r")
yeatsClustData = list(csv.DictReader(file, delimiter=","))
file.close()

file = open("orRes.csv", "r")
orClustData = list(csv.DictReader(file, delimiter=","))
file.close()

print(yeatsClustData[1]['embNo'])


# Prepare environment
run_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
output_folder = Path(f"outputs/{run_id}")
output_folder.mkdir(parents=True, exist_ok=True)
log_file = output_folder / "log.txt"

# Load pre-trained GPT-2 model and tokenizer
#model = GPT2LMHeadModel.from_pretrained('gpt2')
#tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#model.eval()
#model.to("cpu")

# Load pre-trained Sentence Transformer model
# Nomic is a newer model with dialable embedding dimensions
#matryoshka_dim = 384
#model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

#embedYeats = model.encode(yeats, convert_to_tensor=True)
#embedYeats = F.layer_norm(embedYeats, normalized_shape=(embedYeats.shape[1],))
#embedYeats = embedYeats[:, :matryoshka_dim]
#embedYeats = F.normalize(embedYeats, p=2, dim=1)

#embedOreilly = model.encode(oreilly, convert_to_tensor=True)
#embedOreilly = F.layer_norm(embedOreilly, normalized_shape=(embedOreilly.shape[1],))
#embedOreilly = embedOreilly[:, :matryoshka_dim]
#embedOreilly = F.normalize(embedOreilly, p=2, dim=1)

#embedYeats = embedYeats.numpy()
#embedOreilly = embedOreilly.numpy()

# all-MiniLM-L6-v2 option
model = SentenceTransformer("all-MiniLM-L6-v2")
embedYeats = model.encode(yeats)
embedOreilly = model.encode(oreilly)

print(embedYeats.shape)
print(embedOreilly.shape)

#verse structure (for table)
verseCols = [["palegoldenrod"],["palegoldenrod"],["palegoldenrod"],["palegoldenrod"],["palegoldenrod"],["palegoldenrod"],["palegoldenrod"],["palegoldenrod"],
             ["lavender"],["lavender"],["lavender"],["lavender"],["lavender"],["lavender"],["lavender"],["lavender"],
             ["thistle"],["thistle"],["thistle"],["thistle"],["thistle"],["thistle"],["thistle"],["thistle"],
             ["lightsteelblue"],["lightsteelblue"],["lightsteelblue"],["lightsteelblue"],["lightsteelblue"],["lightsteelblue"],["lightsteelblue"],["lightsteelblue"],
             ["moccasin"],["moccasin"],["moccasin"],["moccasin"],["moccasin"],["moccasin"],["moccasin"],["moccasin"]]

embedYeatsMagMax = np.max(embedYeats)
embedYeatsMagMin = np.min(embedYeats)
embedOreillyMagMax = np.max(embedOreilly)
embedOreillyMagMin = np.min(embedOreilly)

print(embedYeatsMagMax)
print(embedYeatsMagMin)
print(embedOreillyMagMax)
print(embedOreillyMagMin)



# Initialize logging
with open(log_file, 'w') as f:
  f.write(f"Run ID: {run_id}\nText: {yeats}\nOutputs:\n")

# Create Power Spectra
wavelet = 'morl'
scales = np.arange(1, 32)
images = []
yeatsPowerSpectrum = []
oreillyPowerSpectrum = []

for i in range(len(embedYeats[1])):
  coef, freq = pywt.cwt(embedYeats[:, i], scales, wavelet)
  power = (abs(coef))**2
  yeatsPowerSpectrum.append(power)

for i in range(len(embedOreilly[1])):
  coef, freq = pywt.cwt(embedOreilly[:, i], scales, wavelet)
  power = (abs(coef))**2
  oreillyPowerSpectrum.append(power)

yeatsPowerMax = np.max(yeatsPowerSpectrum)
oreillyPowerMax = np.max(oreillyPowerSpectrum)

#df_yeatsEmb = pd.DataFrame(embedYeats)
#df_oreillyEmb = pd.DataFrame(embedOreilly)

#annMat_yeatsPowSpec = []
#for x in range(len(yeatsPowerSpectrum)):
#  for y in range(len(yeatsPowerSpectrum[x])):
#      for z in range(len(yeatsPowerSpectrum[x][y])):
#        annMat_yeatsPowSpec.append([x, y, z, yeatsPowerSpectrum[x][y][z]])

#df_yeatsPowSpec = pd.DataFrame(annMat_yeatsPowSpec, columns=['embeddingdim', 'period', 'line', 'value'])
#df_yeatsPowSpec.to_csv(output_folder / f"df_yeatsPowSpec.csv", index=False)

#annMat_oreillyPowSpec = []
#for x in range(len(oreillyPowerSpectrum)):
#  for y in range(len(oreillyPowerSpectrum[x])):
#      for z in range(len(oreillyPowerSpectrum[x][y])):
#        annMat_oreillyPowSpec.append([x, y, z, oreillyPowerSpectrum[x][y][z]])

#df_oreillyPowSpec = pd.DataFrame(annMat_oreillyPowSpec, columns=['embeddingdim', 'period', 'line', 'value'])
#df_oreillyPowSpec.to_csv(output_folder / f"df_oreillyPowSpec.csv", index=False)

#df_yeatsEmb.to_csv(output_folder / f"df_yeatsEmb.csv")
#df_oreillyEmb.to_csv(output_folder / f"df_oreillyEmb.csv")




# Plot & Save Power Spectra
for i in range(len(yeatsPowerSpectrum)):

  embIndex = int(yeatsClustData[i]['embNo'])
  
  fig = plt.figure(figsize=(12, 8), layout="compressed", dpi=200)
  ax_dict = fig.subplot_mosaic(
      [[
          "power", "power", "power", "power", "power", "power", "power", "power", "power", "magnitude", "magnitude", "magnitude"
      ],
       [
           "power", "power", "power", "power", "power", "power", "power", "power", "power", "magnitude", "magnitude", "magnitude"
       ],
       [
           "power", "power", "power", "power", "power", "power", "power", "power", "power", "magnitude", "magnitude", "magnitude"
       ],
       [
           "power", "power", "power", "power", "power", "power", "power", "power", "power", "magnitude", "magnitude", "magnitude"
       ],
       [
           "power", "power", "power", "power", "power", "power", "power", "power", "power", "magnitude", "magnitude", "magnitude"
       ],
       [
           "power", "power", "power", "power", "power", "power", "power", "power", "power", "magnitude", "magnitude", "magnitude"
       ],
       [
           "power", "power", "power", "power", "power", "power", "power", "power", "power", "magnitude", "magnitude", "magnitude"
       ],
       [
           "power", "power", "power", "power", "power", "power", "power", "power", "power", "magnitude", "magnitude", "magnitude"
       ],
       [
           "power", "power", "power", "power", "power", "power", "power", "power", "power", "magnitude", "magnitude", "magnitude"
       ],
       [
           "power", "power", "power", "power", "power", "power", "power", "power", "power", "magnitude", "magnitude", "magnitude"
       ],
       [
           "power", "power", "power", "power", "power", "power", "power", "power", "power", "magnitude", "magnitude", "magnitude"
       ],
       [
           "power", "power", "power", "power", "power", "power", "power", "power", "power", "magnitude", "magnitude", "magnitude"
       ]], )
  
  fig.suptitle('Embedding Dimension ' + str(embIndex))
  ax_dict["power"].tick_params(color=(1,1,1,0), labelcolor=(1,1,1,0))
  #ax_dict["power"] = ax_dict["power"].twinx()
  ax_dict["power"] = ax_dict["power"].twiny()
  ax_dict["power"].set_title('Power Spectrum')
  #powerScale = ax_dict["power"].imshow(yeatsPowerSpectrum[i], cmap='coolwarm', vmin=0, vmax=yeatsPowerMax, aspect='auto')
  powerScale = ax_dict["power"].imshow(np.rot90(yeatsPowerSpectrum[embIndex-1], k = 3), cmap='coolwarm', vmin=0, vmax=yeatsPowerMax, aspect='auto', extent=[len(scales), 0, len(embedYeats[:,embIndex-1]), 0])
  #ax_dict["power"].set_ylabel('Line of Poem')
  ax_dict["power"].set_xlabel('Period')
  #ax_dict["power"].set_yticks(np.arange(0, len(embedYeats[:, i]), 10), np.arange(0, len(embedYeats[:, i]), 10))
  #ax_dict["power"].get_yaxis().set_visible(False)
  ax_dict["power"].set_xticks(np.arange(0, len(scales), 4), scales[::4])
  poemTable = ax_dict["power"].table(cellText=[[line] for line in yeats], loc='right', cellLoc='left', edges = 'closed', colWidths=[0.42], cellColours=verseCols)
  for c in poemTable.properties()['celld'].values():
    c.set(linewidth=0)
  poemTable.scale(1,0.925)
  poemTable.auto_set_font_size(False)
  poemTable.set_fontsize(6)

  base = ax_dict["magnitude"].transData
  rot = transforms.Affine2D().rotate_deg(270)
  
  ax_dict["magnitude"].set_title('Dimension Mag')
  ax_dict["magnitude"].plot(embedYeats[:, embIndex-1], transform = rot + base, marker='o', markersize = 4)
  ax_dict["magnitude"].grid(axis = 'y', which = 'both', color='black', linestyle='-', linewidth=1)
  ax_dict["magnitude"].set_ymargin(0.01)
  ax_dict["magnitude"].set_ylabel('Line of Poem')
  ax_dict["magnitude"].get_yaxis().set_visible(False)
  ax_dict["magnitude"].set_xlabel('Magnitude')
  ax_dict["magnitude"].set_xlim(embedYeatsMagMin, embedYeatsMagMax)
  #ax_dict["magnitude"].set_ylim(0, len(embedYeats[:, i]))
  

  fig.colorbar(ax=ax_dict["power"],
               mappable=powerScale,
               label="Wavelet Power",
               location="left")

  img_path = output_folder / f"power_spectrum_yeats_{i + 1}.png"
  fig.savefig(img_path)
  plt.close(fig=None)
  images.append(img_path)

  with open(log_file, 'a') as f:
    f.write(f"Generated {img_path}\n")


'''
for i in range(len(oreillyPowerSpectrum)):

  fig = plt.figure(figsize=(12, 8), layout="constrained", dpi=150)
  ax_dict = fig.subplot_mosaic(
      [[
          "power", "power", "power", "power", "power", "power", "power",
          "power", "power", "power", "power", "power"
      ],
       [
           "power", "power", "power", "power", "power", "power", "power",
           "power", "power", "power", "power", "power"
       ],
       [
           "power", "power", "power", "power", "power", "power", "power",
           "power", "power", "power", "power", "power"
       ],
       [
           "power", "power", "power", "power", "power", "power", "power",
           "power", "power", "power", "power", "power"
       ],
       [
           "power", "power", "power", "power", "power", "power", "power",
           "power", "power", "power", "power", "power"
       ],
       [
           "power", "power", "power", "power", "power", "power", "power",
           "power", "power", "power", "power", "power"
       ],
       [
           "power", "power", "power", "power", "power", "power", "power",
           "power", "power", "power", "power", "power"
       ],
       [
           "power", "power", "power", "power", "power", "power", "power",
           "power", "power", "power", "power", "power"
       ],
       [
           "power", "power", "power", "power", "power", "power", "power",
           "power", "power", "power", "power", "power"
       ],
       [
           "power", "power", "power", "power", "power", "power", "power",
           "power", "power", "power", "power", "power"
       ],
       [
           "magnitude", "magnitude", "magnitude", "magnitude", "magnitude",
           "magnitude", "magnitude", "magnitude", "magnitude", "magnitude",
           "magnitude", "magnitude"
       ],
       [
           "magnitude", "magnitude", "magnitude", "magnitude", "magnitude",
           "magnitude", "magnitude", "magnitude", "magnitude", "magnitude",
           "magnitude", "magnitude"
       ]], )

  fig.suptitle('Extract from OReilly Graph Databases, Embedding Dimension ' +
               str(i + 1))
  ax_dict["power"].set_title('Power Spectrum')
  powerScale = ax_dict["power"].imshow(numpy.rot90(oreillyPowerSpectrum[i],
                                                   k=1,
                                                   axes=(1, 0)),
                                       cmap='coolwarm',
                                       vmin=0,
                                       vmax=oreillyPowerMax,
                                       aspect='auto')
  ax_dict["power"].set_ylabel('Line of Extract')
  ax_dict["power"].set_xlabel('Period')
  ax_dict["power"].set_yticks(np.arange(0, len(embedOreilly[:, i]), 10),
                              np.arange(0, len(embedOreilly[:, i]), 10))
  ax_dict["power"].set_xticks(np.arange(0, len(scales), 4), scales[::4])

  ax_dict["magnitude"].set_title('Embedding Dimension Value')
  ax_dict["magnitude"].plot(embedOreilly[:, i])
  ax_dict["magnitude"].set_xlabel('Line of Poem')
  ax_dict["magnitude"].set_ylabel('Magnitude')
  ax_dict["magnitude"].set_ylim(embedOreillyMagMin, embedOreillyMagMax)

  fig.colorbar(ax=ax_dict["power"], mappable=powerScale, label="Wavelet Power")

  img_path = output_folder / f"power_spectrum_oreilly_{i+1}.png"
  fig.savefig(img_path)
  plt.close(fig=None)
  images.append(img_path)

  with open(log_file, 'a') as f:
    f.write(f"Generated {img_path}\n")

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

'''
