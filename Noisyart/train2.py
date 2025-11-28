from ultralytics import YOLO
import torch
import os

def run_training():

    DATA_DIR = "dataset"
    PROJECT_DIR = os.path.dirname(__file__)

    MODEL = "yolo11l-cls.pt"

    IMG_SIZE = 448
    EPOCHS = 120
    BATCH = 16
    PATIENCE = 30

    DEVICE = 0 if torch.cuda.is_available() else "cpu"
    print(f" Treinando no dispositivo: {DEVICE}")

    model = YOLO(MODEL)

    print("\n Iniciando treinamento YOLOv11-CLS OTIMIZADO...\n")

    model.train(
        data=DATA_DIR,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        patience=PATIENCE,
        workers=16,
        amp=True,

        dropout=0.05,
        mixup=0.1,
        scale=0.2,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,

        optimizer="AdamW",
        lr0=4e-4,
        momentum=0.9,
        weight_decay=0.0004,

        project=PROJECT_DIR,
        name="runs_optimized"
    )

    print("\nTreinamento finalizado!")
    print(" Melhor modelo: runs_optimized/weights/best.pt")


if __name__ == "__main__":
    run_training()
