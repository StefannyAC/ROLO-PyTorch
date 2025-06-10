# PROYECTO ROLO: COMPUTER VISION I 2025

# Este script prepara un dataset personalizado para entrenar un modelo LSTM que predice las coordenadas de los cuadros delimitadores de objetos en secuencias de video, de acuerdo al experimento 3.
# Recordemos que el experimento 3 consiste en usar todos los frames y solo 1/3 de los GT's disponibles para train y todos para test (evaluación).
# Requiere PyTorch (torch==2.2.2+cu118, torchaudio==2.2.2+cu118, torchvision==0.17.2+cu118), NumPy (numpy==1.26.3), tqdm (tqdm==4.65.2) y OpenCV (opencv-python==4.11.0.86) instalados.
# Adicionalmente, se necesita el extractor de características YOLOv8, que se encuentra implementado en el archivo YOLOv8FeatureExtractor.py.

# Realizado por: Luis Angel Rivas y Stefanny Arboleda

import os # Para manipulación de rutas y archivos
import numpy as np # Para manejo de datos numéricos
import cv2 # Para manipulación de imágenes
from tqdm import tqdm # Barra de progreso para iteraciones
from YOLOv8FeatureExtractor import YOLOv8FeatureExtractor # Extractor de características con YOLOv8
import re # Para expresiones regulares, usado para procesar el ground truth

# Ruta al archivo que contiene la lista de nombres de los videos del dataset OTB30/OTB100 (subconjunto de OTB100)
# Para la creación de este dataset, nos hemos basado en el paper original de ROLO, para poder comparar resultados.
# El archivo debe contener los nombres de los videos, uno por línea (es el mismo archivo usado para todos lo experimentos).
list_file = "OTB100/otb30_list.txt" 

# Función para leer el archivo de lista de videos y devolver una lista con los nombres limpios.
# Cada nombre de video corresponde a una secuencia de video en el dataset OTB100.
def get_video_names():
    """
    Lee el archivo de lista de videos y devuelve una lista con los nombres limpios.

    Returns:
        list[str]: Lista de nombres de carpetas de videos (cada una corresponde a una secuencia de video).
    """
    with open(list_file, 'r') as f:
        videos = [v.strip() for v in f if v.strip()]   
    return videos

# Configuración
otb30videos = get_video_names()  # Lista de nombres de videos

otb30_videos = otb30videos[:30]  # Primeros 30 videos para entrenamiento y test
otb30_eval_videos = otb30videos[30:] # Últimos videos para evaluación (opcional), solo para comprobar generalización (no incluidos en el paper original)

base_path = "OTB100" # Directorio raíz del dataset
save_path_train = "data/ROLO_data_exp3/train" # Carpeta donde se guardarán los datos de entrenamiento
os.makedirs(save_path_train, exist_ok=True) # Crear carpeta de entrenamiento si no existe

extractor = YOLOv8FeatureExtractor(model_path='yolov8s.pt') # Inicializar el extractor de características YOLOv8 (version small)

def process_videos_metodo3(video_list, base_path, save_path, extractor, label_fraction=1/3):
    """
    Preprocesa una lista de videos extrayendo características con YOLOv8 y generando
    secuencias con etiquetado parcial o completo, según se indique.

    Args:
    -----------
    video_list : list of str
        Lista de nombres de videos a procesar (carpetas con imágenes y ground truth).
    base_path : str
        Ruta base donde están los videos y groundtruth_rect.txt.
    save_path : str
        Ruta donde se guardarán los .npy con features, GT y predicciones.
    extractor : YOLOv8FeatureExtractor
        Instancia del extractor de características que devuelve vector + bbox.
    label_fraction : float
        Fracción de frames que se etiquetarán con ground truth (por defecto 1/3).

    Returns:
    --------
    None. Guarda arrays numpy en la carpeta correspondiente por video:
        - features.npy: vectores de características por frame.
        - gt.npy: ground truths normalizados (parciales si label_fraction < 1).
        - bbox_yolo.npy: bboxes predichos por YOLOv8 normalizados.
    """
    for video in video_list:
        print(f"Procesando video: {video}")
        img_dir = os.path.join(base_path, video, "img") # Carpeta con imágenes del video 
        gt_file = os.path.join(base_path, video, "groundtruth_rect.txt") # Archivo con ground truth del video
        output_dir = os.path.join(save_path, video) # Carpeta donde se guardarán los datos procesados
        os.makedirs(output_dir, exist_ok=True) # Crear carpeta de salida si no existe

        # Leer ground truth
        gts = []
        with open(gt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = re.split(r'[, \t]+', line) # Usar expresiones regulares para dividir por comas, espacios o tabulaciones pues algunos archivos tienen diferentes formatos
                # Asegurarse de que hay al menos 4 partes (x, y, w, h)
                # Esto es importante porque algunos archivos pueden tener líneas con menos datos
                # o con formato diferente, y queremos evitar errores al convertir a float (no observado este comportamiento pero es mejor prevenirlo).
                if len(parts) >= 4:
                    gts.append(list(map(float, parts[:4]))) # [x, y, w, h] x, y, width, height

        # Obtener lista de imágenes del video y emparejar con GT
        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
        min_len = min(len(gts), len(img_files))
        if len(gts) != len(img_files):
            print(f"[AVISO] {video} tiene {len(img_files)} imágenes, {len(gts)} GTs.")
            img_files = img_files[:min_len]
            gts = gts[:min_len]

        print(f"{video}: usando {min_len} frames. GT parcial 1:{int(1/label_fraction)})") # Visualización de cuántos frames se usarán y la fracción de GTs etiquetados

        features, gts_normalized, bboxes_yolo = [], [], [] # Features extraídos por YOLOv8, ground truth normalizado y bounding boxes estimados por YOLOv8s

        for i, fname in enumerate(tqdm(img_files, desc=f"Procesando {video}")):
            img_path = os.path.join(img_dir, fname)
            original_image = cv2.imread(img_path)
            if original_image is None:
                continue

            orig_h, orig_w = original_image.shape[:2]

            # Reescalar imagen a 640x640 para el extractor de características
            # Esto es necesario porque el extractor espera imágenes de este tamaño
            image = cv2.resize(original_image, (640, 640))
            scale_x = 640 / orig_w
            scale_y = 640 / orig_h

            # Extraer características y bounding box con YOLOv8s
            bbox_pred, vector = extractor.extract(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            features.append(vector)
            bboxes_yolo.append(bbox_pred) # ya normalizado respecto a 640

            # Etiquetado parcial: solo una fracción de los frames tendrá GT
            # Si label_fraction es 1, se etiquetan todos los frames (como lo necesitamos para test y evaluación)
            # Si label_fraction es 1/3, se etiquetan 1 de cada 3 frames
            if i % int(1 / label_fraction) == 0:

                # Normalizar GT respecto a 640x640
                x, y, bw, bh = gts[i]
                x *= scale_x
                y *= scale_y
                bw *= scale_x
                bh *= scale_y

                # Normalizar GT a [0, 1] respecto a 640x640
                # x_center, y_center son las coordenadas del centro del bounding box
                x_center = (x + bw / 2) / 640
                y_center = (y + bh / 2) / 640
                w_n = bw / 640
                h_n = bh / 640
                gts_normalized.append([x_center, y_center, w_n, h_n])
            else:
                gts_normalized.append([-1, -1, -1, -1])  # Marcador de "sin etiqueta"

        # Guardar
        np.save(os.path.join(output_dir, "features.npy"), np.array(features))
        np.save(os.path.join(output_dir, "gt.npy"), np.array(gts_normalized))
        np.save(os.path.join(output_dir, "bbox_yolo.npy"), np.array(bboxes_yolo))
        print(f"{video}: {len(features)} frames procesados con GT parcial.")

# Procesar videos para entrenamiento
# Usamos 1/3 de los GTs disponibles, como se indica en el experimento 3.
process_videos_metodo3(otb30_videos, base_path, save_path_train, extractor, label_fraction=1/3)

print("Procesamiento de videos de entrenamiento completado.\n")

# Procesar videos para test
# Usamos todos los GTs disponibles, como se indica en el experimento 3.
# Esto es necesario para evaluar el modelo en todos los frames.
save_path_test = "data/ROLO_data_exp3/test" # Carpeta donde se guardarán los datos de test
os.makedirs(save_path_test, exist_ok=True) # Crear carpeta de test si no existe
process_videos_metodo3(otb30_videos, base_path, save_path_test, extractor, label_fraction=1)

print("Procesamiento de videos de test completado.\n")

# Procesar videos de evaluación
save_path_eval = "data/ROLO_data_exp3/eval" # Carpeta donde se guardarán los datos de evaluación
os.makedirs(save_path_eval, exist_ok=True) # Crear carpeta de evaluación si no existe
process_videos_metodo3(otb30_eval_videos, base_path, save_path_eval, extractor, label_fraction=1)

print("Procesamiento de videos de evaluación completado.\n")
