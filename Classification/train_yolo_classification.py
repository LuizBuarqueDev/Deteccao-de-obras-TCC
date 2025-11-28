from ultralytics import YOLO
import torch


def run_training():
    DATASET_PATH = "dataset"

    # Modelo máximo (mais preciso)
    MODEL_NAME = "yolov8x-cls.pt"

    IMAGE_SIZE = 224
    EPOCHS = 100
    PATIENCE = 30

    # batch automático baseado na VRAM da RTX 3060
    BATCH = 64  # seguro e rápido

    DEVICE = 0 if torch.cuda.is_available() else "cpu"
    print("🔥 GPU detectada:", DEVICE)

    print("📦 Carregando modelo:", MODEL_NAME)
    model = YOLO(MODEL_NAME)

    print("🚀 Iniciando treino otimizado...")

    results = model.train(
        data=DATASET_PATH,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH,
        device=DEVICE,
        workers=0,           # Windows precisa disso
        patience=PATIENCE,   # early stopping
        amp=True,            # mixed precision (rápido e estável)
        augment=True,        # augment leve e automático
        verbose=True,
    )

    print("\n🎉 Treinamento finalizado!")
    print("📁 Modelo salvo em: runs/classify/train/weights/best.pt")


if __name__ == "__main__":
    run_training()