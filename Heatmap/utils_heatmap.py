import numpy as np

class HeatmapHelper:
    def __init__(self, grid_size=32):
        self.grid_size = grid_size

    def loc_to_coordinates(self, loc):
        """
        Convierte un bounding box normalizado [x_center, y_center, w, h] en
        coordenadas de esquina (x1, y1, x2, y2) en el grid del heatmap.
        """
        loc = [i * self.grid_size for i in loc]  # Escala al grid
        x1 = int(loc[0] - loc[2] / 2)
        y1 = int(loc[1] - loc[3] / 2)
        x2 = int(loc[0] + loc[2] / 2)
        y2 = int(loc[1] + loc[3] / 2)

        # Asegurarse de que esté dentro de los límites
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(self.grid_size - 1, x2), min(self.grid_size - 1, y2)
        return [x1, y1, x2, y2]

    def coordinates_to_heatmap_vec(self, coord):
        """
        Dado un bounding box en coordenadas (x1, y1, x2, y2), crea el heatmap plano (vector de 1024).
        """
        heatmap_vec = np.zeros(self.grid_size * self.grid_size)
        x1, y1, x2, y2 = coord
        for y in range(y1, y2 + 1):
            for x in range(x1, x2 + 1):
                index = y * self.grid_size + x
                heatmap_vec[index] = 1.0
        return heatmap_vec

    def loc_to_heatmap_vec(self, loc):
        """
        Convierte directamente una caja normalizada [x, y, w, h] a un vector heatmap.
        """
        if loc[0] < 0 or loc[1] < 0:
            return np.zeros(self.grid_size * self.grid_size)  # sin GT
        coord = self.loc_to_coordinates(loc)
        return self.coordinates_to_heatmap_vec(coord)

    def heatmap_vec_to_2d(self, heatmap_vec):
        """
        Convierte vector 1D de 1024 a matriz 2D (32x32).
        """
        return heatmap_vec.reshape((self.grid_size, self.grid_size))
