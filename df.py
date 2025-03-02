import pandas as pd
import json
import re

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
        return data

path = '/mnt/c/Users/meduq/OneDrive/Escritorio/redes_dataset_1.json'

data = read_json(path)
df = pd.DataFrame(data)
df.to_csv('data.csv', index=False)

links = df['link']
with open('links.txt', 'w') as f:
    for link in links:
        f.write(link + '\n')