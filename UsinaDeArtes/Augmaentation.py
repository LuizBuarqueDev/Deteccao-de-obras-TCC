import os
import cv2
import albumentations as A
from multiprocessing import Pool, cpu_count
from glob import glob
from tqdm import tqdm
import shutil
import random

# ------------------------------
# CONFIGURAÇÕES
# ------------------------------
INPUT_DIR = "dataset-original"     # originais
TEMP_DIR = "_temp_aug"             # onde ficará tudo antes do split
OUTPUT_DIR = "dataset"             # dataset final

TARGET = 30
IMG_SIZE = 384

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.20
TEST_SPLIT = 0.10


# ------------------------------
# PIPELINE DE AUGMENTATION
# ------------------------------
augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    A.MotionBlur(blur_limit=3, p=0.3),
    A.Rotate(limit=10, border_mode=cv2.BORDER_REFLECT101, p=0.7),
    A.RandomResizedCrop(
        size=(IMG_SIZE, IMG_SIZE),
        scale=(0.85, 1.0),
        p=0.6
    ),
    A.CLAHE(clip_limit=2.0, p=0.3),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=10, p=0.7),
    A.Perspective(scale=(0.02, 0.05), p=0.3),
])


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def augment_class(class_name):
    """
    Copia originais → TEMP
    Gera augmentations até TOTAL = TARGET
    (Split ainda não acontece aqui)
    """
    class_in = os.path.join(INPUT_DIR, class_name)
    class_temp = os.path.join(TEMP_DIR, class_name)
    ensure_dir(class_temp)

    originals = glob(class_in + "/*.jpg") + glob(class_in + "/*.png") + glob(class_in + "/*.jpeg")
    originals = sorted(originals)

    # Copia todas as originais
    for img_path in originals:
        shutil.copy2(img_path, os.path.join(class_temp, os.path.basename(img_path)))

    current = len(originals)

    if current < TARGET:
        need = TARGET - current
        print(f"[AUG] Classe {class_name}: gerando {need} images...")

        idx = current
        for i in tqdm(range(need), desc=class_name):
            base_img = originals[i % current]
            img = cv2.imread(base_img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            aug = augment(image=img)["image"]
            aug = cv2.cvtColor(aug, cv2.COLOR_RGB2BGR)

            out_path = os.path.join(class_temp, f"aug_{idx}.jpg")
            cv2.imwrite(out_path, aug)
            idx += 1

    return f"{class_name}: total={TARGET}"


def split_dataset():
    print("\n📌 Realizando divisão train/val/test...")

    classes = sorted(os.listdir(TEMP_DIR))

    for class_name in classes:
        class_temp = os.path.join(TEMP_DIR, class_name)

        images = (
            glob(class_temp + "/*.jpg") +
            glob(class_temp + "/*.png") +
            glob(class_temp + "/*.jpeg")
        )
        random.shuffle(images)

        n = len(images)
        n_train = int(n * TRAIN_SPLIT)
        n_val = int(n * VAL_SPLIT)
        n_test = n - n_train - n_val

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train+n_val],
            "test": images[n_train+n_val:]
        }

        for split_name, imgs in splits.items():
            out_dir = os.path.join(OUTPUT_DIR, split_name, class_name)
            ensure_dir(out_dir)

            for img_path in imgs:
                shutil.copy2(img_path, os.path.join(out_dir, os.path.basename(img_path)))

        print(f"✔ {class_name}: train={n_train}, val={n_val}, test={n_test}")


def main():

    print("🚀 AUGMENTAÇÃO → SPLIT")

    ensure_dir(TEMP_DIR)
    ensure_dir(OUTPUT_DIR)

    classes = sorted(os.listdir(INPUT_DIR))
    workers = min(20, cpu_count())

    # 1) AUGMENTAÇÃO COMPLETA
    print("\n🔧 Gerando aumentações...")
    with Pool(workers) as pool:
        results = pool.map(augment_class, classes)
    for r in results:
        print(r)

    # 2) SPLIT
    split_dataset()

    print("\n🎉 FINALIZADO COM SUCESSO!")
    print(f"📁 Dataset final em: {OUTPUT_DIR}/train, val, test")


if __name__ == "__main__":
    main()
