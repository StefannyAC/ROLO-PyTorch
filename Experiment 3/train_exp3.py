# PROYECTO ROLO: COMPUTER VISION I 2025

# Este script entrena un modelo LSTM para predecir las coordenadas de los cuadros delimitadores de objetos en secuencias de video, para el experimento 3.
# Recordemos que el experimento 3 consiste en usar todos los frames y solo 1/3 de sus GT's disponibles para train y todos para test (evaluación).
# Requiere PyTorch (torch==2.2.2+cu118, torchaudio==2.2.2+cu118, torchvision==0.17.2+cu118) y NumPy (numpy==1.26.3) instalados.
# Antes de ejecutar este script, se debe haber preparado el dataset de ROLO con las características y bounding boxes normalizados a través
# del script `dataloader_exp3.py`, que es el experimento que se quiere realizar.
# En la base de datos OTB100, donde se han usado los 30 videos listados en 'OTB100/otb30_list.txt'.

# Realizado por: Luis Angel Rivas y Stefanny Arboleda

import os # Para manipulación de rutas y archivos
import numpy as np # Para manejo de datos numéricos
import torch # Biblioteca principal de PyTorch
from torch import nn, optim # Módulos de redes neuronales y optimización
from torch.utils.data import Dataset, DataLoader # Para manejo de datasets y batchs

num_epochs = 100 # Número de épocas para el entrenamiento
sequence_length = 6 # Longitud de la secuencia para el modelo LSTM

# Clase para organizar el Dataset personalizado
class ROLOConcatDataset(Dataset):
    """Dataset personalizado para el modelo ROLO (experimento 3) con secuencias de características y cajas YOLO.
    Args:
        base_dir (str): Ruta al directorio que contiene los datos de entrenamiento organizados por video.
        sequence_length (int): Longitud de la secuencia temporal para alimentar al LSTM.
    Attributes:
        samples (list): Lista de tuplas (input_sequence, gt_sequence) para todas las secuencias válidas encontradas.
    Returns:
        Cada item del dataset es una tupla (x, y):
            x: Tensor de entrada de tamaño (sequence_length, 516) (512 características extraidas con Yolov8s + 4 coordenadas de bbox)
            y: Tensor de salida (ground truth) de tamaño (sequence_length, 4)
    """
    def __init__(self, base_dir, sequence_length):
        self.sequence_length = sequence_length
        self.samples = [] # Lista donde se almacenan las secuencias generadas

        # Iteramos sobre los videos (carpetas) dentro del directorio base
        for video in sorted(os.listdir(base_dir)):
            path = os.path.join(base_dir, video)
            if not os.path.isdir(path):
                continue

            # Rutas a los archivos necesarios previamente generados con el dataloader pertinente
            # features.npy: características extraídas con YOLOv8
            # bbox_yolo.npy: coordenadas de los bounding boxes normalizados
            # gt.npy: ground truth de las coordenadas de los bounding boxes
            # Se asume que estos archivos están organizados por video en carpetas separadas
            features_path = os.path.join(path, "features.npy")
            bbox_path = os.path.join(path, "bbox_yolo.npy")
            gt_path = os.path.join(path, "gt.npy")

            if not (os.path.exists(features_path) and os.path.exists(bbox_path) and os.path.exists(gt_path)):
                continue

            # Carga los datos desde los archivos .npy
            features = np.load(features_path)
            bboxes = np.load(bbox_path)
            gt = np.load(gt_path)

            # Nos aseguramos de que todas las secuencias tengan la misma longitud
            min_len = min(len(features), len(bboxes), len(gt))
            features, bboxes, gt = features[:min_len], bboxes[:min_len], gt[:min_len]

            # Concatenamos las features y las cajas para formar el input final (T, 516) , que ingresa al LSTM
            # features: (T, 512) + bboxes: (T, 4) -> inputs: (T, 516)
            # Aquí, 512 son las características extraídas por YOLOv8 y 4 son las coordenadas del bounding box
            # Normalmente, las coordenadas del bounding box son [x, y, w, h], donde:
            # x, y son las coordenadas de la esquina superior izquierda del bounding box,
            # w es el ancho y h es la altura del bounding box.
            # En este caso, se asume que las coordenadas ya están normalizadas y listas para ser utilizadas.
            inputs = np.concatenate([features, bboxes], axis=1)

            # Dividimos los datos en secuencias de longitud fija
            for i in range(0, len(inputs) - sequence_length):
                input_seq = inputs[i:i+sequence_length] # Secuencia de entrada
                gt_seq = gt[i:i+sequence_length] # Secuencia de ground truth
                self.samples.append((input_seq, gt_seq)) # Añadimos al conjunto de entrenamiento

    def __len__(self):
        """
        Devuelve la cantidad total de muestras en el dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Devuelve la muestra correspondiente al índice dado.

        Args:
            idx (int): Índice de la muestra.

        Returns:
            x (Tensor): Secuencia de entrada, shape (sequence_length, 516)
            y (Tensor): Secuencia de salida esperada (ground truth), shape (sequence_length, 4)
        """
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Modelo LSTM
class ROLO_LSTM(nn.Module):
    """
    Modelo ROLO basado en LSTM para predicción de cajas de seguimiento.

    Args:
        input_size (int): Tamaño del vector de entrada en cada instante de tiempo (default=516).
        hidden_size (int): Tamaño del estado oculto del LSTM (default=512).
        output_size (int): Tamaño de la salida por paso de tiempo (default=4, para coordenadas de bbox).
        num_layers (int): Número de capas LSTM (default=1).

    Returns:
        Para una entrada de tamaño (batch, sequence_length, input_size), retorna una salida de
        tamaño (batch, sequence_length, output_size).
    """
    def __init__(self, input_size=516, hidden_size=512, output_size=4, num_layers=1):
        super().__init__()
        # Capa LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Capa totalmente conectada para transformar hidden state en coordenadas de bbox
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Propagación hacia adelante del modelo.

        Args:
            x (Tensor): Tensor de entrada de tamaño (batch, seq_len, input_size)

        Returns:
            Tensor de salida de tamaño (batch, seq_len, output_size)
        """
        out, _ = self.lstm(x) # out tiene tamaño (batch, seq_len, hidden_size)
        return self.fc(out) # se proyecta a (batch, seq_len, 4)

# Entrenamiento
def train():
    """
    Función principal de entrenamiento para el modelo ROLO-LSTM.
    Carga los datos, entrena el modelo durante un número fijo de épocas y guarda el modelo final.
    """
    # Ruta a los datos de entrenamiento donde se hayan guardado despues de ejecutar el dataloader_exp3.py
    base_dir = "data/ROLO_data_exp3/train"
    dataset = ROLOConcatDataset(base_dir, sequence_length=sequence_length) # Inicializa dataset
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True) # Carga en batches
    output_dir = "checkpoints" # Directorio donde se guardará el modelo
    os.makedirs(output_dir, exist_ok=True) # Crea el directorio si no existe

    # Usa GPU si está disponible, en este caso CUDA, en una NVIDIA RTX Ti Super 4070 (16GB)
    # Si no, usa CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Usando dispositivo:", device)

    # Inicializamos modelo, optimizador y función de pérdida de acuerdo a la arquitectura del ROLO-LSTM (del paper original)
    model = ROLO_LSTM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Dado que en este caso el entrenamiento se realiza con una secuencia de frames y solo 1/3 de sus GT's disponibles,
    # usamos una función de pérdida MSE con reducción 'none' para poder aplicar una máscara que ignore los GT's no disponibles.
    criterion = nn.MSELoss(reduction='none')  # Para poder aplicar máscara usamos MSE con reducción 'none'

    model.train() # Establece el modelo en modo entrenamiento

    # Ciclo principal de entrenamiento
    for epoch in range(num_epochs):
        total_loss = 0.0
        for inputs, targets in dataloader:
            inputs = inputs.to(device)          # (1, seq_len, 516)
            targets = targets.to(device)        # (1, seq_len, 4)

            outputs = model(inputs)             # (1, seq_len, 4)

            # Crear máscara: True donde targets son válidos
            # El ground-truth fue rellenado con -1 donde no hay información disponible
            # Por lo tanto, si todas las coordenadas del bbox son distintas de -1, el frame es válido
            mask = (targets != -1).all(dim=2)   # (1, seq_len)

            # Calcular pérdida por elemento
            loss_matrix = criterion(outputs, targets)  # (1, seq_len, 4)

            # Aplicar máscara y promediar solo las pérdidas válidas
            valid_losses = loss_matrix[mask]    # vector (num_valid, )
            # Si hay pérdidas válidas, las promediamos. Si no, devolvemos 0.0 (sin afectar entrenamiento)
            loss = valid_losses.mean() if valid_losses.numel() > 0 else torch.tensor(0.0, device=device)

            optimizer.zero_grad() # Reinicia los gradientes para el optimizador
            # Backward pass y optimización
            loss.backward()
            optimizer.step() # Actualiza los pesos del modelo

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        print(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.6f}")

    # Guardar el modelo entrenado
    torch.save(model.state_dict(), os.path.join(output_dir, f"rolo_lstm_exp3_{num_epochs}.pth"))
    print(f"Modelo guardado como {output_dir}/rolo_lstm_exp3_{num_epochs}.pth")

if __name__ == "__main__":
    train()
