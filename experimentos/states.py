import numpy as np
import matplotlib.pyplot as plt

class MiniGridEnv:
    def __init__(self):
        self.grid_size = 13  
    
    def get_reachable_positions(self, start_x, start_y, steps=1):
        """Genera las posiciones únicas alcanzables desde una posición dada en un número de pasos."""
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        reachable_positions = set()
        
        def dfs(x, y, remaining_steps):
            if remaining_steps == 0:
                reachable_positions.add((x, y))
                return
            for dx, dy in moves:
                nx, ny = x + dx, y + dy
                if 1 <= nx <= 11 and 1 <= ny <= 11:  # Asumiendo límites del tablero
                    dfs(nx, ny, remaining_steps - 1)
        
        dfs(start_x, start_y, steps)
        return reachable_positions

def contar_posiciones_alcanzables_tablero(env, num_pasos):
    """
    Cuenta las posiciones únicas alcanzables desde cada celda en el tablero y las almacena en una matriz.
    """
    reachable_positions_grid = np.zeros((env.grid_size, env.grid_size))
    
    for x in range(1, env.grid_size - 1):
        for y in range(1, env.grid_size - 1):
            reachable_positions = env.get_reachable_positions(x, y, num_pasos)
            reachable_positions_grid[x, y] = len(reachable_positions)
    
    return reachable_positions_grid


env = MiniGridEnv()
num_pasos = 6 
reachable_positions_grid = contar_posiciones_alcanzables_tablero(env, num_pasos)

plt.figure(figsize=(8, 8))
plt.imshow(reachable_positions_grid, cmap="plasma", origin="upper")
plt.colorbar(label="Cantidad de Posiciones Alcanzables")
plt.title(f"Heatmap de Posiciones Alcanzables en el Tablero a {num_pasos} Pasos")
plt.xlabel("Posición X")
plt.ylabel("Posición Y")
plt.show()
