import pandas as pd
import json

json_path = '/mnt/c/Users/meduq/OneDrive/Escritorio/redes_dataset_1.json'

df = pd.read_json(json_path)
df.to_csv('data.csv', index=False)