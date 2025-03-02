import requests
from bs4 import BeautifulSoup
import json
import re
import os
from urllib.parse import urlparse

# Lista de URLs de las páginas de inmuebles en renta
with open('links.txt', 'r') as file:
    urls = [f"{line.strip()}"  for line in file.readlines()]

# Lista para almacenar la información de cada inmueble
inmuebles = []

output_folder = "json_data"
os.makedirs(output_folder, exist_ok=True)

for url in urls:
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        # Buscar el script que contiene el JSON
        next_data_script = soup.find("script", id="__NEXT_DATA__", type="application/json")
        if next_data_script:
            try:
                data = json.loads(next_data_script.string)
            except Exception as e:
                print(f"Error parseando JSON en {url}: {e}")
                continue
            # Crear un nombre de archivo usando la última parte de la URL
            path = urlparse(url).path
            identifier = path.strip("/").replace("/", "_")
            filename = os.path.join(output_folder, f"{identifier}.json")
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"Datos guardados en {filename}")
        else:
            print(f"No se encontró el script __NEXT_DATA__ en {url}")
    else:
        print(f"Error al acceder a {url} (Status Code: {response.status_code})")

print("Extracción completada.")
