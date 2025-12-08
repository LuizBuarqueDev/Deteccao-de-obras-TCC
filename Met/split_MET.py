import os
import csv
import re
import shutil
import random
import unicodedata
from collections import defaultdict

TXT_PATH = "MetObjects.txt"
SOURCE = "MET"
DEST = "dataset"
MAX_LEN = 180

TRAIN = 0.7
VAL = 0.2
TEST = 0.1

VALID_EXT = {".jpg", ".jpeg", ".png"}
TARGET_CLASSES = 100

# Função de sanitização
def sanitize(name):
    if not name:
        return None
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r'[\u200b-\u200f\u2028\u2029\u0000-\u001f\u007f-\u009f]', '', name)
    name = re.sub(r'[<>:"/\\\\|?*\[\]\(\)\{\}]', '_', name)
    name = name.replace("／", "_").replace("＼", "_")
    name = re.sub(r'\.+$', '', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip(" _")
    name = name[:MAX_LEN].strip()
    if not name or re.fullmatch(r'[_\W]+', name):
        return None
    return name

# Carregar IDs existentes
existing_ids = {d for d in os.listdir(SOURCE) if d.isdigit()}
print(f"Classes encontradas em MET/: {len(existing_ids)}")

# Mapear objeto -> título
def load_title_map():
    id_to_title = {}
    dup_count = defaultdict(int)
    with open(TXT_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=",", quotechar='"')
        next(reader)
        for row in reader:
            if len(row) < 10:
                continue
            object_id = row[4].strip()
            title = row[9].strip()
            if object_id not in existing_ids:
                continue
            clean = sanitize(title)
            if not clean:
                continue
            if clean in dup_count:
                dup_count[clean] += 1
                clean = f"{clean} ({dup_count[clean]})"
            else:
                dup_count[clean] = 1
            id_to_title[object_id] = clean
    print(f"Títulos válidos carregados: {len(id_to_title)} classes")
    return id_to_title

TITLE_MAP = load_title_map()

# Listar imagens
def list_images(path):
    return [f for f in os.listdir(path) if os.path.splitext(f)[1].lower() in VALID_EXT]

# Split por classe
classes_created = 0

for class_id in existing_ids:
    if classes_created >= TARGET_CLASSES:
        break
    if class_id not in TITLE_MAP:
        continue

    class_name = TITLE_MAP[class_id]
    src = os.path.join(SOURCE, class_id)
    imgs = list_images(src)
    if len(imgs) < 3:
        continue

    random.shuffle(imgs)
    n = len(imgs)
    n_train = max(int(n * TRAIN), 1)
    n_val   = max(int(n * VAL), 1)
    n_test  = n - n_train - n_val
    if n_test < 1:
        n_test = 1
        n_train -= 1

    train_imgs = imgs[:n_train]
    val_imgs   = imgs[n_train:n_train+n_val]
    test_imgs  = imgs[n_train+n_val:]

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(DEST, split, class_name), exist_ok=True)

    for img in train_imgs:
        shutil.copy2(os.path.join(src, img), os.path.join(DEST, "train", class_name, img))
    for img in val_imgs:
        shutil.copy2(os.path.join(src, img), os.path.join(DEST, "val", class_name, img))
    for img in test_imgs:
        shutil.copy2(os.path.join(src, img), os.path.join(DEST, "test", class_name, img))

    classes_created += 1

print(f"\nProcesso finalizado! Total de classes criadas: {classes_created}")
