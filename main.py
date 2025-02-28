#This is now a script for preprocessing texts into embeddings, power spectra etc.

import datetime
import json
import csv
import os
from pysondb import db
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from matplotlib import transforms
from sentence_transformers import SentenceTransformer
import psycopg2.pool

def build_DB():

  cur.execute(
    """CREATE TABLE text_header (
        text_id SERIAL PRIMARY KEY,
        textTitle VARCHAR(255),
        textType VARCHAR(20),
        textSplitLevel VARCHAR(20),
        embeddingModel VARCHAR(50),
        splitLevelDims INT,
        embeddingDims INT          
    )
    """)
  cur.execute(
    """
    CREATE TABLE text_parts (
        text_id SERIAL REFERENCES text_header (text_id),
        text_part_seq INT PRIMARY KEY,
        text_part text        
    )
    """)
  cur.execute(
    """
    CREATE TABLE embeddings (
        text_id SERIAL REFERENCES text_header (text_id),
        embed_dim_id INT PRIMARY KEY,
        embedding_series FLOAT []
    )
    """)
  cur.execute(
    """
    CREATE TABLE wavPowerSpectrum (
        text_id SERIAL REFERENCES text_header (text_id),
        embed_dim_id INT REFERENCES embeddings (embed_dim_id),
        power_spectrum FLOAT [][]
    )
    """)
  conn.commit()


pool = psycopg2.pool.SimpleConnectionPool(0, 80, os.environ['DATABASE_URL'])
conn = pool.getconn()

cur = conn.cursor()

build_DB()

cur.close()
pool.putconn(conn)

pool.closeall()



'''
DROP SCHEMA public CASCADE;
CREATE SCHEMA public;
GRANT ALL ON SCHEMA public TO public;
'''
  

'''
embedText = []
powerSpectrum = []

with open('./wAIvGUI/texts/gatsby.json', 'r') as f:
  content = json.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")
embedText = model.encode(content['para'])
embedTextSave = embedText.tolist()

embeddingDict = dict(textTitle = "The Great Gatsby", textType = "Book", level = "para", embeddingModel = "all-MiniLM-L6-v2", levelDims = 1609, embeddingDims = 384, embeddings = embedTextSave)


if os.path.exists("./wAIvGUI/processedTexts/GatsbyEmbed.json"):
  os.remove("./wAIvGUI/processedTexts/GatsbyEmbed.json")
else:
  print("The file does not exist")

with open("./wAIvGUI/processedTexts/GatsbyEmbed.json", "w") as outfile:
  json.dump(embeddingDict, outfile, indent=4, sort_keys=False)

wavelet = 'morl'
scaleMax = int(np.ceil(len(embedText) * 1.5))
print(scaleMax)
scales = np.arange(1, scaleMax)



wpDB = db.getDb("./wAIvGUI/processedTexts/GatsbyWavPower.json")

for i in range(len(embedText[1])):
  power, freq = pywt.cwt(embedText[:, i], scales, wavelet)
  power = (abs(power))**2

  wavPowDict = dict(textTitle = "The Great Gatsby", textType = "Book", level = "para", embeddingModel = "all-MiniLM-L6-v2", levelDims = 1609, embeddingDims = 384, scale = scaleMax, waveletType = "Morlet", powSpecID = i, waveletPower = power.tolist())
  
  wpDB.add(wavPowDict)
  #powerSpectrum.append(power.tolist())
  print(i)

'''

#if os.path.exists("./wAIvGUI/processedTexts/GatsbyWavPower.json"):
#  os.remove("./wAIvGUI/processedTexts/GatsbyWavPower.json")
#else:
#  print("The file does not exist")
#
#with open("./wAIvGUI/processedTexts/GatsbyWavPower.json", "w") as outfile:
#  json.dump(wavPowDict, outfile, indent=4, sort_keys=False)