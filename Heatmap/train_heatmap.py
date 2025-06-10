# === Entrenamiento para ROLO con heatmap (Experimento 3) ===

import os                                       # Para manejar rutas y crear carpetas.
import torch                                    # Librería principal de deep learning.
from torch import nn, optim                     # Para funciones de pérdida y optimizadores.
from torch.utils.data import DataLoader         # Para manejo de lotes de datos.
from HeatmapDataset_exp3 import ROLOHeatmapDataset, ROLO_LSTM_Heatmap  # Dataset y modelo definidos específicamente para el experimento 3.


num_epochs = 100                # Número de épocas para entrenar el modelo.
sequence_length = 6            # Longitud de la secuencia temporal que recibe la LSTM.


# === Entrenamiento ===
def train():
    base_dir = "data/ROLO_data_exp3/train"                         # Directorio donde están los videos preprocesados para entrenamiento.
    dataset = ROLOHeatmapDataset(base_dir, sequence_length=sequence_length)  # Carga el dataset personalizado con heatmaps.
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)             # DataLoader con lotes de tamaño 1 y mezcla aleatoria.


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Usa GPU si está disponible, de lo contrario CPU.
    print("Usando dispositivo:", device)


    model = ROLO_LSTM_Heatmap().to(device)        # Modelo LSTM modificado para salida con sigmoid.
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Optimizador Adam con tasa de aprendizaje baja.
    # Función de pérdida binaria clásica (ya que la salida del modelo aplica sigmoid). 
    # Compara directamente los valores de probabilidad predichos contra los valores esperados binarios (0 o 1).
    criterion = nn.MSELoss()


    model.train()   # Establece el modelo en modo entrenamiento.
    for epoch in range(num_epochs):
        total_loss = 0.0
        for inputs, targets in dataloader:
            inputs = inputs.to(device)     # Entrada de forma (1, 6, 1536) — secuencia de 6 pasos con vectores concatenados.
            targets = targets.to(device)   # Ground truth heatmap aplanado: (1, 1024).


            outputs = model(inputs)        # Salida del modelo: (1, 1024), valores entre 0 y 1 gracias a sigmoid.
            loss = criterion(outputs, targets)  # Cálculo de la pérdida entre heatmaps predicho y ground truth.


            optimizer.zero_grad()  # Limpia gradientes acumulados.
            loss.backward()        # Backpropagation.
            optimizer.step()       # Actualiza los pesos.
            total_loss += loss.item()  # Suma la pérdida de esta iteración para el promedio.

        avg_loss = total_loss / len(dataloader) # Pérdida promedio por época.
        print(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.6f}")

    # Crea carpeta de checkpoints si no existe y guarda el modelo entrenado.
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/rolo_lstm_heatmap_exp3_{num_epochs}.pth")
    print(f"✅ Modelo guardado como checkpoints/rolo_lstm_heatmap_exp3_{num_epochs}.pth")

# Llama la función de entrenamiento cuando se ejecuta el script directamente.
if __name__ == "__main__":
    train()
