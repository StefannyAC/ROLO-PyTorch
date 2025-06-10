# PROYECTO ROLO: COMPUTER VISION I 2025

"""Para cada video, se crean secuencias de longitud fija combinando features y bboxes, convertido en este caso a un vector
de 1024 dimensiones que corresponde a hacer un flatten del grid de 32x32=1024, y se utiliza el último bounding box del ground truth como objetivo
mapeado de la misma manera a un vector heatmap. Luego, un modelo ROLO_LSTM_Heatmap toma cada secuencia, la procesa con una LSTM y predice el mapa 
de calor correspondiente al último paso temporal, el cual indica la probabilidad espacial de la ubicación futura del objeto."""

# Recordemos que el experimento 3 consiste en usar todos los frames y solo 1/3 de sus GT's disponibles para train y todos para test (evaluación).
# Requiere PyTorch (torch==2.2.2+cu118, torchaudio==2.2.2+cu118, torchvision==0.17.2+cu118) y NumPy (numpy==1.26.3) instalados.
# En la base de datos OTB100, donde se han usado los 30 videos listados en 'OTB100/otb30_list.txt'.

# Realizado por: Luis Angel Rivas y Stefanny Arboleda


import os # Para manipulación de rutas y archivos.
import numpy as np # Para manipulación de datos numéricos.
import torch # Biblioteca principal de PyTorch.
from torch.utils.data import Dataset 
from utils_heatmap import HeatmapHelper  # Clase que convierte un bounding box en un vector heatmap.

class ROLOHeatmapDataset(Dataset):

    # Inicializar la longitud de la secuencia y un contenedor samples.
    def __init__(self, base_dir, sequence_length=3):
        self.sequence_length = sequence_length
        self.samples = []
        self.heatmap_helper = HeatmapHelper(grid_size=32) # Se usará para convertir coordenadas de bounding boxes a vectores heatmap.

        # Itera sobre todos los subdirectorios (uno por video) en base_dir.
        for video in sorted(os.listdir(base_dir)):
            path = os.path.join(base_dir, video)
            if not os.path.isdir(path):
                continue
            
            # Construye las rutas para los archivos .npy esperados en cada video: características, bbox predicho (YOLO), y ground truth (gt).
            features_path = os.path.join(path, "features.npy")
            bbox_path = os.path.join(path, "bbox_yolo.npy")
            gt_path = os.path.join(path, "gt.npy")
            # Ignorar si falta alguno
            if not (os.path.exists(features_path) and os.path.exists(bbox_path) and os.path.exists(gt_path)):
                continue
            # Carga los archivos .npy
            features = np.load(features_path)
            bboxes = np.load(bbox_path)
            gt = np.load(gt_path)

            # Recorta todos los arreglos a la misma longitud mínima (para evitar problemas de sincronización).
            min_len = min(len(features), len(bboxes), len(gt))
            features, bboxes, gt = features[:min_len], bboxes[:min_len], gt[:min_len]

            # Concatenar features con bbox (ambos normalizados ya) y convertidos a Vector Heatmap
            bboxes_heatmap = np.array([self.heatmap_helper.loc_to_heatmap_vec(b) for b in bboxes])
            inputs = np.concatenate([features, bboxes_heatmap], axis=1)  # (T, 512 + 1024)

            # Crea muestras por ventanas deslizantes de longitud sequence_length.
            for i in range(0, len(inputs) - sequence_length + 1):
                input_seq = inputs[i:i+sequence_length]  # shape: (seq_len, 1536)
                gt_box = gt[i+sequence_length-1]  # última bbox del bloque

                # Convertir GT a heatmap
                if np.any(gt_box < 0):  # ignorar si no hay etiqueta
                    continue
                heatmap_vec = self.heatmap_helper.loc_to_heatmap_vec(gt_box)

                # Guarda el par (input, heatmap) en la lista de muestras.
                self.samples.append((input_seq, heatmap_vec))

    # Devuelve cuántas muestras hay en el dataset.
    def __len__(self):
        return len(self.samples)

    # Convierte cada muestra a tensores de PyTorch cuando es solicitada por un DataLoader.
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

#  Crea una red LSTM simple que toma entradas de tamaño 1536, 
#  tiene un tamaño oculto 512, y produce salidas de dimensión 1024 (un vector flatten del mapa de calor 32×32).
class ROLO_LSTM_Heatmap(torch.nn.Module):
    def __init__(self, input_size=1536, hidden_size=512, output_size=1024, num_layers=1):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])   # logits
        return torch.sigmoid(out)     # aplicar sigmoide explícitamente