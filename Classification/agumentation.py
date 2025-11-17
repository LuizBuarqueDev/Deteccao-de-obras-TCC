import os
import cv2
import albumentations as A
import multiprocessing
from functools import partial

# CONFIGURAÇÕES
input_folder = "dataset"
output_folder = "output"
AUG_PER_IMAGE = 30  # aumente se quiser mais variações

# TRANSFORMAÇÕES COMBINADAS (melhor resultado)
transform = A.Compose([
    A.Rotate(limit=20, p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.2, p=0.7),
    A.GaussNoise(p=0.4),
    A.Blur(blur_limit=3, p=0.3),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.4),
    A.Affine(scale=(1.0, 1.3), p=0.5),
], p=1)

os.makedirs(output_folder, exist_ok=True)


def process_image(cls, file):
    cls_path = os.path.join(input_folder, cls)
    cls_out = os.path.join(output_folder, cls)

    os.makedirs(cls_out, exist_ok=True)

    img_path = os.path.join(cls_path, file)
    img = cv2.imread(img_path)

    if img is None:
        print("Erro ao carregar:", img_path)
        return

    filename = os.path.splitext(file)[0]

    # Salva imagem original
    cv2.imwrite(os.path.join(cls_out, f"{filename}_original.jpg"), img)

    # Gera augmentations
    for i in range(AUG_PER_IMAGE):
        augmented = transform(image=img)["image"]
        out_name = f"{filename}_aug{i}.jpg"
        out_path = os.path.join(cls_out, out_name)
        cv2.imwrite(out_path, augmented)


if __name__ == "__main__":
    print("\n🚀 Iniciando geração paralela...\n")

    tasks = []

    for cls in os.listdir(input_folder):
        cls_path = os.path.join(input_folder, cls)
        if not os.path.isdir(cls_path):
            continue

        files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        print(f"📁 Classe {cls}: {len(files)} imagens")

        for file in files:
            tasks.append((cls, file))

    # Usa todos os núcleos da CPU
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    # Executar as tasks em paralelo
    pool.starmap(process_image, tasks)

    pool.close()
    pool.join()

    print("\n🎉 Finalizado! Arquivos gerados em:", output_folder)
