import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = "MET"
JSON_PATH = "MET_database.json"
MIN_IMAGES = 11

with open(JSON_PATH, "r") as f:
    data = json.load(f)

ids_validos = {str(item["id"]) for item in data}

todas_classes = [
    d for d in os.listdir(BASE_DIR)
    if os.path.isdir(os.path.join(BASE_DIR, d))
]

def processar_classe(classe):
    class_path = os.path.join(BASE_DIR, classe)

    if classe not in ids_validos:
        shutil.rmtree(class_path)
        return f"DEL (não no JSON): {classe}"

    files = os.listdir(class_path)
    images = [
        f for f in files
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if len(images) < MIN_IMAGES:
        shutil.rmtree(class_path)
        return f"DEL (Com menos de {MIN_IMAGES} imagens): {classe}"

    return f"OK: {classe}"

results = []

with ThreadPoolExecutor(max_workers=20) as executor:
    futures = {
        executor.submit(processar_classe, c): c
        for c in todas_classes
    }
    for future in as_completed(futures):
        results.append(future.result())

print("\n".join(results))
print("\n Finalizado! Todas as classes fora do JSON e todas com 1 foto foram removidas.")
