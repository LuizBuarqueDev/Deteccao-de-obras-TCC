from ultralytics import YOLO
import torch
import os

def run_training():

    DATA_DIR = "dataset"

    PROJECT_DIR = os.path.dirname(__file__)

    MODEL = "yolo11m-cls.pt"

    IMG_SIZE = 384
    EPOCHS = 50
    BATCH = 32
    PATIENCE = 20

    DEVICE = 0 if torch.cuda.is_available() else "cpu"
    print(f" Treinando no dispositivo: {DEVICE}")

    model = YOLO(MODEL)

    print("\n Iniciando treinamento YOLOv11 CLASSIFIER...\n")

    model.train(
        data=DATA_DIR,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        patience=PATIENCE,
        workers=16,
        amp=True,
        dropout=0,    
        verbose=True,

        project= PROJECT_DIR,
        name="runs"
    )

    print("\n🎉 Treinamento finalizado!")
    print("📁 Melhor modelo: runs/classify/train*/weights/best.pt")


if __name__ == "__main__":
    run_training()
