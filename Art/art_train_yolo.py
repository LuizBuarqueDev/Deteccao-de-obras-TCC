from ultralytics import YOLO
import torch

def run_training():

    DATA_CONFIG = "data.yaml"
    MODEL_NAME = "yolo11m.pt"

    IMAGE_SIZE = 640
    EPOCHS = 200
    PATIENCE = 50

    BATCH = 50

    DEVICE = 0 if torch.cuda.is_available() else "cpu"
    print(f" Dispositivo detectado: {DEVICE}")

    print(f" Carregando modelo {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)

    print(" Iniciando treinamento YOLOv11 DETECTION...")

    model.train(
        data=DATA_CONFIG,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH,
        device=DEVICE,
        workers=0,
        patience=PATIENCE,
        amp=True,
        cache=False,

        lr0=0.01,
        warmup_epochs=3,
        verbose=True,

        
    project=r"C:\Users\gabri\github\Deteccao-de-obras-TCC\Art",
    name="treino_art",
    )

    print("\n Treinamento finalizado!")
    print(" Melhor modelo salvo em: runs/detect/train/weights/best.pt")


if __name__ == "__main__":
    run_training()