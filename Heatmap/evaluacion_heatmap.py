# PROYECTO ROLO: COMPUTER VISION I 2025

"""En este script se carga el modelo ya entrenado ROLO_LSTM_Heatmap, predice un heatmap por cada una de las secuencias
y se hace uan visualización de los resultados"""

# Recordemos que el experimento 3 consiste en usar todos los frames y solo 1/3 de sus GT's disponibles para train y todos para test (evaluación).
# Requiere PyTorch (torch==2.2.2+cu118, torchaudio==2.2.2+cu118, torchvision==0.17.2+cu118) y NumPy (numpy==1.26.3) instalados.
# En la base de datos OTB100, donde se han usado los 30 videos listados en 'OTB100/otb30_list.txt'.

# Realizado por: Luis Angel Rivas y Stefanny Arboleda
import os
import cv2
import numpy as np
import torch
from HeatmapDataset_exp3 import ROLO_LSTM_Heatmap  # Modelo entrenado
from utils_heatmap import HeatmapHelper            # Utilidad para convertir bbox ↔ heatmap

# Configuración 
num_epochs = 100  
sequence_length = 6
base_dir = "data/ROLO_data_exp3/test"  # Carpeta con los datos de test o eval.
model_path = f"checkpoints/rolo_lstm_heatmap_exp3_{num_epochs}.pth" # Cargar los pesos entrenados
output_root = f"evaluations_heatmap_exp3_{num_epochs}" # Carpeta donde se guardan los resultados
os.makedirs(output_root, exist_ok=True)

# Modelo en GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ROLO_LSTM_Heatmap().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  # Poner en modo evaluación

# Utilidad para convertir bboxes a y desde vectores heatmap
helper = HeatmapHelper(grid_size=32)

# Convertir vector heatmap a bounding box (centro estimado + tamaño fijo) ===
def heatmap_to_bbox(heatmap_vec, default_wh=(0.1, 0.1)):
    heatmap = torch.sigmoid(heatmap_vec).view(32, 32).cpu().numpy()
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    x_center = (x + 0.5) / 32.0
    y_center = (y + 0.5) / 32.0
    return [x_center, y_center, *default_wh]

# Calcular IoU entre dos bounding boxes en formato [x_center, y_center, w, h] ===
def calculate_iou(box1, box2):
    # Convertir a [x1, y1, x2, y2]
    box1 = [box1[0] - box1[2]/2, box1[1] - box1[3]/2, box1[0] + box1[2]/2, box1[1] + box1[3]/2]
    box2 = [box2[0] - box2[2]/2, box2[1] - box2[3]/2, box2[0] + box2[2]/2, box2[1] + box2[3]/2]
    
    # Calcular intersección
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0  # No hay intersección
    
    inter_area = (x_right - x_left) * (y_bottom - y_top)
    union_area = ((box1[2] - box1[0]) * (box1[3] - box1[1]) +
                  (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter_area)
    return inter_area / union_area

# Evaluación por video 
video_metrics = {}

for video_name in os.listdir(base_dir):
    video_dir = os.path.join(base_dir, video_name)
    if not os.path.isdir(video_dir):
        continue

    print(f"\nEvaluando video: {video_name}")

    # === Cargar archivos necesarios ===
    features_path = os.path.join(video_dir, "features.npy")
    bbox_path = os.path.join(video_dir, "bbox_yolo.npy")
    gt_path = os.path.join(video_dir, "gt.npy")

    if not (os.path.exists(features_path) and os.path.exists(bbox_path) and os.path.exists(gt_path)):
        print(f"Datos incompletos en {video_name}. Saltando...")
        continue

    features = np.load(features_path)
    bboxes = np.load(bbox_path)
    gt = np.load(gt_path)

    # === Convertir bboxes a heatmaps y concatenar con features ===
    bboxes_heatmap = np.array([helper.loc_to_heatmap_vec(b) for b in bboxes])  # (N, 1024)
    inputs = np.concatenate([features, bboxes_heatmap], axis=1)                # (N, 1536)

    # Carpeta para guardar resultados visuales
    output_dir = os.path.join(output_root, video_name)
    os.makedirs(output_dir, exist_ok=True)
    ious = []

    # Ruta a las imágenes originales del video
    img_dir = f"/home/rivas0806/OTB100/OTB100/{video_name}/img"
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])

    with torch.no_grad():
        for start in range(0, len(inputs) - sequence_length + 1):
            # Preparar secuencia de entrada
            x_seq = torch.tensor(inputs[start:start+sequence_length], dtype=torch.float32).unsqueeze(0).to(device)

            # Predecir heatmap
            pred_vec = model(x_seq).squeeze(0)  # (1024,)
            pred_box = heatmap_to_bbox(pred_vec)  # [x_center, y_center, w, h] fijo
            gt_box = gt[start + sequence_length - 1]  # bbox real
            iou = calculate_iou(pred_box, gt_box)
            ious.append(iou)

            idx = start + sequence_length - 1
            if idx >= len(img_files):
                break

            # Cargar imagen del frame
            img_path = os.path.join(img_dir, img_files[idx])
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]

            # Dibujar Ground Truth (verde) 
            x, y, bw, bh = gt_box
            x1, y1 = int((x - bw/2) * w), int((y - bh/2) * h)
            x2, y2 = int((x + bw/2) * w), int((y + bh/2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Generar imagen de heatmap
            heatmap = torch.sigmoid(pred_vec).view(32, 32).cpu().numpy()
            heatmap_resized = cv2.resize(heatmap, (w, h))
            heatmap_resized -= heatmap_resized.min()
            heatmap_resized /= heatmap_resized.max() + 1e-6  # Normalizar

            heatmap_uint8 = np.uint8(255 * heatmap_resized)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

            # Guardar frame resultante 
            cv2.imwrite(os.path.join(output_dir, f"{idx:04d}.jpg"), overlay)

    # Guardar métricas del video
    avg_iou = np.mean(ious) if ious else 0.0
    video_metrics[video_name] = avg_iou
    print(f"{video_name}: IoU promedio = {avg_iou:.4f}")

# Reporte global
print("\n=== Resumen de métricas ===")
for video, iou in video_metrics.items():
    print(f"{video}: {iou:.4f}")

global_avg_iou = np.mean(list(video_metrics.values())) if video_metrics else 0.0
print(f"\nIoU promedio global: {global_avg_iou:.4f}")

# Guardar métricas en archivo
with open(os.path.join(output_root, "metrics.txt"), "w") as f:
    f.write("Video\tAvg IoU\n")
    for video, iou in video_metrics.items():
        f.write(f"{video}\t{iou:.4f}\n")
    f.write(f"\nGlobal\t{global_avg_iou:.4f}\n")

print(f"\nResultados guardados en: {output_root}")
