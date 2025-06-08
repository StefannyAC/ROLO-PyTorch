# PROYECTO ROLO: COMPUTER VISION I 2025

# Este script implementa un extractor de características utilizando YOLOv8 para extraer características visuales y bounding boxes de imágenes.
# Requiere PyTorch (torch==2.2.2+cu118, torchaudio==2.2.2+cu118, torchvision==0.17.2+cu118), NumPy (numpy==1.26.3), OpenCV (opencv-python==4.11.0.86) y Ultralytics instalados.

# Realizado por: Luis Angel Rivas y Stefanny Arboleda

import torch
from ultralytics import YOLO
import numpy as np
import cv2

class YOLOv8FeatureExtractor:
    """
    Clase para extraer características de imágenes usando un modelo YOLOv8 preentrenado.

    Este extractor realiza dos tareas principales:
    1. Detectar el objeto con mayor confianza en una imagen.
    2. Extraer un vector de características profundo a partir de las capas internas del modelo YOLOv8s, 
        específicamente de la última capa del backbone de YOLOv8s, cuya salida tiene 512 canales de características.
        
    Attributes:
    -----------
    model_path : str
        Ruta al modelo YOLOv8 preentrenado (por defecto: 'yolov8s.pt').
    device : str
        Dispositivo de cómputo ('cuda' si está disponible, de lo contrario 'cpu').
    model : YOLO
        Modelo YOLO cargado y listo para inferencia.
    """
    def __init__(self, model_path='yolov8s.pt', device=None):
        """
        Inicializa el extractor de características cargando el modelo YOLOv8.

        Parameters
        ----------
        model_path : str
            Ruta al archivo .pt del modelo YOLOv8 (por defecto: 'yolov8s.pt').
        device : str or None
            Dispositivo para ejecutar el modelo ('cuda' o 'cpu').
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu') # Usar GPU si está disponible
        self.model = YOLO(model_path).to(self.device) # Cargar el modelo y mover al dispositivo
        self.model.eval() # Poner el modelo en modo evaluación

    def extract(self, image):
        """
        Extrae el cuadro delimitador (bbox) normalizado y el vector de características de la imagen dada.

        Parameters
        ----------
        image : np.ndarray
            Imagen de entrada en formato BGR (como es común con OpenCV), de tamaño arbitrario.

        Returns
        -------
        bbox : np.ndarray, shape=(4,)
            Coordenadas normalizadas del bbox detectado: (x_center, y_center, width, height) en [0, 1].
            Si no se detecta ningún objeto, retorna [0.5, 0.5, 0.0, 0.0].

        vector : np.ndarray, shape=(512,)
            Vector de características extraído del backbone del modelo YOLOv8.
            Si no se detecta ningún objeto, retorna un vector de ceros.
        """
        # Convertir imagen a RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Inferencia
        results = self.model(img_rgb, verbose = False)[0]

        # Verificar si se detectaron objetos
        if results.boxes is None or results.boxes.data is None or len(results.boxes.data) == 0:
            print("No se detectó ningún objeto.")
            return np.array([0.5, 0.5, 0.0, 0.0]) , np.zeros(512)
        
        # Obtener la caja delimitadora con mayor confianza
        best = max(results.boxes.data, key=lambda b: b[4])  # mayor confianza
        x1, y1, x2, y2, conf, cls = best.tolist()

        # Calcular centro y tamaño, y normalizar con respecto a 640x640
        x_center = (x1 + x2) / 2 / 640
        y_center = (y1 + y2) / 2 / 640
        w = (x2 - x1) / 640
        h = (y2 - y1) / 640
        bbox = np.array([x_center, y_center, w, h])

        # Extraer características del backbone de YOLOv8
        tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        tensor = tensor.to(self.device)

        with torch.no_grad(): # Desactivar el cálculo de gradientes
            x = tensor
            # Pasar por las capas del modelo hasta antes de "Concat" o "Detect"
            for i, block in enumerate(self.model.model.model):
                if "Concat" in block.__class__.__name__ or "Detect" in str(block):
                    break
                x = block(x)

            # Promedio adaptativo para reducir a un vector de tamaño fijo
            vector = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).view(-1)
            vector = vector.cpu().numpy()

        return bbox, vector
