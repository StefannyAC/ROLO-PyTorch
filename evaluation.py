# PROYECTO ROLO: COMPUTER VISION I 2025

# Este script evalúa un modelo LSTM entrenado para predecir cuadros delimitadores en secuencias de video, usando un subconjunto de 30 videos del dataset OTB100.
# Requiere PyTorch (torch==2.2.2+cu118, torchaudio==2.2.2+cu118, torchvision==0.17.2+cu118), NumPy (numpy==1.26.3) y OpenCV (opencv-python==4.11.0.86) instalados.
# Ademmás, se necesita el modelo LSTM entrenado, que debe estar guardado en un archivo .pth y los archivos de características y ground truth generados previamente con el dataloader
# correspondiente a cada experimento.
# En este caso, se usa la métrica de Intersection over Union (IoU) para evaluar el rendimiento del modelo en cada video. Y se calcula el AOS (Average Overlap Score) global que corresponde al IoU promedio de toda la evaluación en cada video.

# Realizado por: Luis Angel Rivas y Stefanny Arboleda

import os # Para manipulación de rutas y archivos
import cv2 # Para manipulación de imágenes
import numpy as np # Para manejo de datos numéricos
import torch # Para operaciones con tensores y modelos de PyTorch
from train import ROLO_LSTM # Importar el modelo LSTM entrenado

# Configuración
num_epochs = 100  # Número de épocas para el entrenamiento
base_dir = "data/ROLO_data_exp3/test"  # Carpeta con videos a evaluar en el conjunto de test 
#Si quisieramos evaluar en el conjunto de evaluación, cambiaríamos a "data/ROLO_data_exp3/eval", se debe cambiar según el experimento
sequence_length = 6  # Debe coincidir con el entrenamiento
model_path = f"checkpoints/rolo_lstm_exp3_{num_epochs}.pth" # Ruta al modelo entrenado, se debe cambiar según el experimento
output_root = f"evaluations_exp3_{num_epochs}"  # Carpeta raíz para resultados, se debe cambiar según el experimento

os.makedirs(output_root, exist_ok=True) # Crear carpeta de resultados si no existe

# Dispositivo para PyTorch (GPU si está disponible, sino CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar modelo entrenado
model = ROLO_LSTM().to(device) # Inicializar el modelo LSTM
model.load_state_dict(torch.load(model_path, map_location=device)) # Cargar los pesos del modelo desde el archivo .pth
model.eval() # Establecer el modelo en modo evaluación (desactiva dropout y batch normalization)

# Función para calcular IoU
# Recordemos que el IoU (Intersection over Union) es una métrica que mide la superposición entre dos cuadros delimitadores.
# Se calcula como el área de intersección entre los dos cuadros dividido por el área de unión de ambos cuadros.
# Los cuadros se representan como [x_center, y_center, width, height] (normalizados entre 0 y 1).
def calculate_iou(box1, box2):
    """
    box: [x_center, y_center, width, height] (normalizado)
    Retorna: IoU (float)
    """
    # Convertir a coordenadas de esquinas (x1, y1, x2, y2)
    box1 = [
        box1[0] - box1[2] / 2,  # x1
        box1[1] - box1[3] / 2,   # y1
        box1[0] + box1[2] / 2,   # x2
        box1[1] + box1[3] / 2    # y2
    ]
    box2 = [
        box2[0] - box2[2] / 2,
        box2[1] - box2[3] / 2,
        box2[0] + box2[2] / 2,
        box2[1] + box2[3] / 2
    ]
    
    # Calcular área de intersección
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    if x_right < x_left or y_bottom < y_top: # Condición de no intersección
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top) 
    
    # Área de unión
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]) # Área del primer cuadro
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]) # Área del segundo cuadro
    # Área de unión es la suma de las áreas individuales menos el área de intersección
    union_area = area1 + area2 - intersection_area
    
    return intersection_area / union_area # Retornar el IoU calculado

# Evaluación por video
video_metrics = {} # Diccionario para almacenar métricas por video

for video_name in os.listdir(base_dir):
    video_dir = os.path.join(base_dir, video_name) # Ruta al directorio del video
    if not os.path.isdir(video_dir):
        continue
    
    print(f"\nEvaluando video: {video_name}")
    
    # Cargar datos
    features_path = os.path.join(video_dir, "features.npy") # Ruta a las características extraídas
    bbox_path = os.path.join(video_dir, "bbox_yolo.npy") # Ruta a las bounding boxes predichas por YOLOv8
    gt_path = os.path.join(video_dir, "gt.npy") # Ruta a los ground truths (GTs)
    
    if not (os.path.exists(features_path) and os.path.exists(bbox_path) and os.path.exists(gt_path)):
        print(f"Datos incompletos en {video_name}. Saltando...")
        continue
    
    features = np.load(features_path) # Cargar características extraídas del video
    bbox_yolo = np.load(bbox_path) # Cargar bounding boxes predichas por YOLOv8
    gt = np.load(gt_path) # Cargar ground truths (GTs) del video
    
    # Concatenar características y bbox
    inputs = np.concatenate([features, bbox_yolo], axis=1)
    
    # Carpeta para resultados del video
    output_dir = os.path.join(output_root, video_name)
    os.makedirs(output_dir, exist_ok=True) # Crear carpeta de resultados del video si no existe
    
    # Listas para métricas
    ious = []

    img_dir = f"OTB100/{video_name}/img" # Carpeta con imágenes del video
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")]) # Obtener lista de imágenes del video
    
    with torch.no_grad(): # Desactivar gradientes para evaluación
        # Procesar secuencias de entrada
        for start in range(0, len(inputs) - sequence_length+1):
            x_seq = torch.tensor(
                inputs[start:start + sequence_length], 
                dtype=torch.float32
            ).unsqueeze(0).to(device)
            
            # Realizar predicción
            # x_seq tiene forma (1, sequence_length, num_features)
            pred_seq = model(x_seq).squeeze(0).cpu().numpy()
            # Obtener la secuencia de ground truths correspondiente
            # pred_seq tiene forma (sequence_length, 4) donde 4 son [x_center, y_center, width, height]
            gt_seq = gt[start:start + sequence_length]
            
            for i in range(sequence_length):
                idx = start + i # Índice del frame actual
                if idx >= len(img_files): 
                    break
                
                # Calcular IoU
                iou = calculate_iou(pred_seq[i], gt_seq[i]) # Calcular IoU entre la predicción y el GT
                ious.append(iou) # Añadir IoU a la lista de métricas
                
                # Visualización
                img_path = os.path.join(img_dir, img_files[idx])  # Asume nombres de frame: 0001.jpg, 0002.jpg..
                img = cv2.imread(img_path) # Leer imagen del frame actual
                if img is None:
                    continue

                h, w = img.shape[:2] # Obtener dimensiones de la imagen
                
                # Dibujar GT (verde)
                x, y, bw, bh = gt_seq[i]
                x1, y1 = int((x - bw/2) * w), int((y - bh/2) * h)
                x2, y2 = int((x + bw/2) * w), int((y + bh/2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Dibujar predicción (rojo)
                px, py, pw, ph = pred_seq[i]
                px1, py1 = int((px - pw/2) * w), int((py - ph/2) * h)
                px2, py2 = int((px + pw/2) * w), int((py + ph/2) * h)
                cv2.rectangle(img, (px1, py1), (px2, py2), (0, 0, 255), 2)
                
                # Añadir texto (IoU)
                cv2.putText(
                    img, f"IoU: {iou:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )
                
                cv2.imwrite(os.path.join(output_dir, f"{idx:04d}.jpg"), img)
    
    # Métricas del video
    avg_iou = np.mean(ious) if ious else 0.0
    video_metrics[video_name] = avg_iou
    print(f"{video_name}: IoU promedio = {avg_iou:.4f}")

# Reporte global
print("\n=== Resumen de métricas ===")
for video, iou in video_metrics.items():
    print(f"{video}: {iou:.4f}")

# Calcular IoU promedio global
global_avg_iou = np.mean(list(video_metrics.values())) if video_metrics else 0.0
print(f"\nIoU promedio global: {global_avg_iou:.4f}")

# Guardar métricas en un archivo
with open(os.path.join(output_root, "metrics.txt"), "w") as f:
    f.write("Video\tAvg IoU\n")
    for video, iou in video_metrics.items():
        f.write(f"{video}\t{iou:.4f}\n")
    f.write(f"\nGlobal\t{global_avg_iou:.4f}\n")

print(f"\nResultados guardados en: {output_root}") # Donde se guardan los resultados de la evaluación