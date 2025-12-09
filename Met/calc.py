import os
from ultralytics import YOLO
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from glob import glob

# Caminho correto do modelo
MODEL_PATH = "runs/weights/best.pt"
TEST_DIR = os.path.abspath("dataset/test")
VALID_EXT = {".jpg", ".jpeg", ".png"}

def load_test_images(test_dir):
    images = []
    labels = []
    classes = sorted(os.listdir(test_dir))
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    for cls_name in classes:
        cls_path = os.path.join(test_dir, cls_name)
        for ext in VALID_EXT:
            for img_path in glob(os.path.join(cls_path, f"*{ext}")):
                images.append(img_path)
                labels.append(class_to_idx[cls_name])
    return images, labels, class_to_idx

# Carrega o modelo corretamente
model = YOLO(MODEL_PATH)

images, true_labels, class_to_idx = load_test_images(TEST_DIR)

pred_labels = []

for img_path in images:
    results = model.predict(img_path, imgsz=384, verbose=False)
    pred_class = int(results[0].probs.top1)
    pred_labels.append(pred_class)

accuracy = accuracy_score(true_labels, pred_labels)
precision = precision_score(true_labels, pred_labels, average="macro")
recall = recall_score(true_labels, pred_labels, average="macro")
f1 = f1_score(true_labels, pred_labels, average="macro")

print("===== Métricas de Teste =====")
print(f"Acurácia: {accuracy:.4f}")
print(f"Precision (macro): {precision:.4f}")
print(f"Recall (macro): {recall:.4f}")
print(f"F1-score (macro): {f1:.4f}")