import os
import random
import shutil

SOURCE = "dataset-original - Copia"
DEST = "dataset"
TRAIN = 0.7
VAL = 0.2
TEST = 0.1
VALID_EXT = {".jpg", ".jpeg", ".png"}
TARGET_CLASSES = 100  # número máximo de classes a processar

def list_images(path):
    return [f for f in os.listdir(path) if os.path.splitext(f)[1].lower() in VALID_EXT]

# Pegar todas as pastas de classes
existing_ids = [d for d in os.listdir(SOURCE) if os.path.isdir(os.path.join(SOURCE, d))]
print(f"Classes encontradas: {len(existing_ids)}")

classes_created = 0

for class_id in existing_ids:
    if classes_created >= TARGET_CLASSES:
        break

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

    # Criar pastas
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(DEST, split, class_id), exist_ok=True)

    # Copiar imagens
    for img in train_imgs:
        shutil.copy2(os.path.join(src, img), os.path.join(DEST, "train", class_id, img))
    for img in val_imgs:
        shutil.copy2(os.path.join(src, img), os.path.join(DEST, "val", class_id, img))
    for img in test_imgs:
        shutil.copy2(os.path.join(src, img), os.path.join(DEST, "test", class_id, img))

    classes_created += 1

print(f"\nProcesso finalizado! Total de classes criadas: {classes_created}")