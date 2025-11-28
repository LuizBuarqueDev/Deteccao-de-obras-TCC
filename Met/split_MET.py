import os
import csv
import re
import shutil
import random
import unicodedata
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from threading import Lock


TXT_PATH = "MetObjects.txt"
SOURCE = "MET"
DEST = "dataset"
MAX_LEN = 180
THREADS = 20

TRAIN = 0.7
VAL = 0.2
TEST = 0.1

VALID_EXT = {".jpg", ".jpeg", ".png"}

# ============================
# DEFINIR O LIMITE DE CLASSES
# ============================
TARGET_CLASSES = 400        # <<< COLOQUE AQUI O NÚMERO DESEJADO
classes_created = 0
lock = Lock()               # usado para sincronização entre threads


# ============================
# SANITIZAÇÃO ROBUSTA DE TÍTULOS
# ============================

def sanitize(name):
    if not name:
        return None

    # Normaliza para ASCII
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()

    # Remove caracteres de controle invisíveis
    name = re.sub(r'[\u200b-\u200f\u2028\u2029\u0000-\u001f\u007f-\u009f]', '', name)

    # Remove caracteres problemáticos em nomes de arquivo
    # REMOVE [ ] ( ) { } também
    name = re.sub(r'[<>:"/\\\\|?*\[\]\(\)\{\}]', '_', name)

    # Substitui versões Unicode de barras
    name = name.replace("／", "_").replace("＼", "_")

    # Remove pontos finais e underscores repetidos
    name = re.sub(r'\.+$', '', name)
    name = re.sub(r'_+', '_', name)

    # Limpa espaços
    name = name.strip(" _")

    # Limite de tamanho
    name = name[:MAX_LEN].strip()

    # Evita nomes apenas com underscore ou vazio
    if not name or re.fullmatch(r'[_\W]+', name):
        return None

    return name


# ============================
# CARREGAR IDS EXISTENTES
# ============================

existing_ids = {d for d in os.listdir(SOURCE) if d.isdigit()}
print(f"Classes encontradas em MET/: {len(existing_ids)}")


# ============================
# MAPEAR OBJETO → TÍTULO
# ============================

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


# ============================
# LISTAR IMAGENS
# ============================

def list_images(path):
    return [
        f for f in os.listdir(path)
        if os.path.splitext(f)[1].lower() in VALID_EXT
    ]


# ============================
# SPLIT POR CLASSE COM LIMITE
# ============================

def split_class(class_id):
    global classes_created

    # VERIFICAR LIMITE ANTES DE TUDO
    with lock:
        if classes_created >= TARGET_CLASSES:
            return

    if class_id not in TITLE_MAP:
        return

    class_name = TITLE_MAP[class_id]

    # PASTA JÁ EXISTE → IGNORA
    if (
        os.path.exists(os.path.join(DEST, "train", class_name)) or
        os.path.exists(os.path.join(DEST, "val", class_name)) or
        os.path.exists(os.path.join(DEST, "test", class_name))
    ):
        return

    src = os.path.join(SOURCE, class_id)
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

    # Criar pastas
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(DEST, split, class_name), exist_ok=True)

    # Copiar imagens
    for img in train_imgs:
        shutil.copy2(os.path.join(src, img), os.path.join(DEST, "train", class_name, img))

    for img in val_imgs:
        shutil.copy2(os.path.join(src, img), os.path.join(DEST, "val", class_name, img))

    for img in test_imgs:
        shutil.copy2(os.path.join(src, img), os.path.join(DEST, "test", class_name, img))

    # Incrementar contador
    with lock:
        classes_created += 1

    return class_id


# ============================
# EXECUTAR
# ============================

print(f"Iniciando split com limite de {TARGET_CLASSES} classes...")

with ThreadPoolExecutor(max_workers=THREADS) as ex:
    list(tqdm(ex.map(split_class, existing_ids), total=len(existing_ids)))

print(f"\nProcesso finalizado! Total de classes criadas: {classes_created}")
