import os
import cv2
import albumentations as A
from glob import glob
from tqdm import tqdm
import random

# CONFIGURAÇÕES
INPUT_DIR = "dataset/train"
TARGET = 30
IMG_SIZE = 384

# PIPELINE DE AUGMENTATION (CORRIGIDO)
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


def augment_class(class_name):
    class_path = os.path.join(INPUT_DIR, class_name)

    originals = (
        glob(class_path + "/*.jpg") +
        glob(class_path + "/*.jpeg") +
        glob(class_path + "/*.png")
    )
    originals = sorted(originals)

    current = len(originals)

    # Se a classe não tem imagens, pular
    if current == 0:
        print(f"Classe vazia, ignorando: {class_name}")
        return

    # Se já tiver imagens suficientes, apenas reporta
    if current >= TARGET:
        print(f"{class_name}: OK ({current})")
        return

    need = TARGET - current
    print(f"[AUG] Classe {class_name}: gerando {need} imagens...")

    idx = current
    for i in tqdm(range(need), desc=class_name):
        base_img = originals[i % current]
        img = cv2.imread(base_img)

        if img is None:
            print(f" Erro ao ler imagem: {base_img}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        aug = augment(image=img)["image"]
        aug = cv2.cvtColor(aug, cv2.COLOR_RGB2BGR)

        out_path = os.path.join(class_path, f"aug_{idx}.jpg")
        cv2.imwrite(out_path, aug)
        idx += 1


    idx = current
    for i in tqdm(range(need), desc=class_name):
        base_img = originals[i % current]
        img = cv2.imread(base_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        aug = augment(image=img)["image"]
        aug = cv2.cvtColor(aug, cv2.COLOR_RGB2BGR)

        out_path = os.path.join(class_path, f"aug_{idx}.jpg")
        cv2.imwrite(out_path, aug)

        idx += 1

def main():
    print("AUGMENTAÇÃO DIRETA NA PASTA TRAIN\n")
    classes = sorted(os.listdir(INPUT_DIR))

    for c in classes:
        augment_class(c)

    print("\nAUGMENTAÇÃO FINALIZADA!")

if __name__ == "__main__":
    main()