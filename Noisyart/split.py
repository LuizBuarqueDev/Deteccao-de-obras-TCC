import os
import csv
import re
import shutil
import random
import unicodedata
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


SOURCE = "trainval_3120"
DEST = "dataset"
MAX_LEN = 120
THREADS = 20

TRAIN = 0.7
VAL = 0.2
TEST = 0.1

VALID_EXT = {".jpg", ".jpeg", ".png"}


def sanitize(name):
    if not name:
        return "Unknown"

    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    name = name[:180].strip()

    return name or "Unknown"



existing_ids = [d for d in os.listdir(SOURCE) if os.path.isdir(os.path.join(SOURCE, d))]
print(f"📂 Classes encontradas em MET/: {len(existing_ids)}")


def list_images(path):
    return [
        f for f in os.listdir(path)
        if os.path.splitext(f)[1].lower() in VALID_EXT
    ]


def split_class(class_name):
    src = os.path.join(SOURCE, class_name)
    imgs = list_images(src)

    if len(imgs) < 3:
        return

    random.shuffle(imgs)
    n = len(imgs)

    n_train = int(n * TRAIN)
    n_val   = int(n * VAL)
    n_test  = n - n_train - n_val

    if n_train < 1: n_train = 1
    if n_val < 1:   n_val = 1
    if n_test < 1:
        n_test = 1
        n_train -= 1

    train_imgs = imgs[:n_train]
    val_imgs   = imgs[n_train:n_train+n_val]
    test_imgs  = imgs[n_train+n_val:]
    clean = sanitize(class_name)

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(DEST, split, clean), exist_ok=True)

    for img in train_imgs:
        shutil.copy2(os.path.join(src, img), os.path.join(DEST, "train", clean, img))

    for img in val_imgs:
        shutil.copy2(os.path.join(src, img), os.path.join(DEST, "val", clean, img))

    for img in test_imgs:
        shutil.copy2(os.path.join(src, img), os.path.join(DEST, "test", clean, img))

    return class_name



print(" Iniciando split paralelo...")

with ThreadPoolExecutor(max_workers=THREADS) as ex:
    list(tqdm(ex.map(split_class, existing_ids), total=len(existing_ids)))

print("\n Split finalizado com sucesso!")
