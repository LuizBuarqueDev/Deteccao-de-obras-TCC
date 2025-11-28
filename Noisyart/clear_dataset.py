import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = "trainval_3120"
MIN_IMAGES = 35   # mínimo de imagens para manter a classe
THREADS = 20      # número de threads

# Todas as classes dentro de MET/
todas_classes = [
    d for d in os.listdir(BASE_DIR)
    if os.path.isdir(os.path.join(BASE_DIR, d))
]

def processar_classe(classe):
    class_path = os.path.join(BASE_DIR, classe)

    # contar imagens
    images = [
        f for f in os.listdir(class_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    # deletar se tiver menos do mínimo
    if len(images) < MIN_IMAGES:
        shutil.rmtree(class_path)
        return f"DEL ({len(images)} imagens — mínimo é {MIN_IMAGES}): {classe}"

    return f"OK: {classe} ({len(images)} imagens)"


results = []

# Paralelo
with ThreadPoolExecutor(max_workers=THREADS) as executor:
    futures = {
        executor.submit(processar_classe, c): c
        for c in todas_classes
    }
    for future in as_completed(futures):
        results.append(future.result())

# Log final
print("\n".join(results))
print(f"\n🏁 Finalizado! Todas as classes com menos de {MIN_IMAGES} imagens foram removidas.")