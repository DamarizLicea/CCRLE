from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall
from minigrid.minigrid_env import MiniGridEnv
import numpy as np
import random
import time

import matplotlib.pyplot as plt
import collections

class MiniGridEnv:
    def __init__(self):
        self.quadrants = [(1, 1), (5, 5), (9, 9), (3, 3)]
        self.current_quadrant = self.quadrants[0]
        self.grid_size = 13  # Tamaño del tablero (13x13)
    
    def encode_state(self, x, y):
        quadrant_index = self.quadrants.index(self.current_quadrant)
        return (x * 100 + y) + quadrant_index
    
    def get_reachable_states(self, start_x, start_y, steps=1):
        """Genera los estados alcanzables desde una posición dada en un número de pasos."""
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        reachable_states = set()
        
        def dfs(x, y, remaining_steps):
            if remaining_steps == 0:
                state = self.encode_state(x, y)
                reachable_states.add(state)
                return
            for dx, dy in moves:
                nx, ny = x + dx, y + dy
                if 1 <= nx <= 11 and 1 <= ny <= 11: 
                    dfs(nx, ny, remaining_steps - 1)
        
        dfs(start_x, start_y, steps)
        return reachable_states

def calcular_empowerment(env, start_x, start_y, num_pasos):
    """
    Calcula el empowerment en función de los estados únicos alcanzables desde la posición inicial
    en una cantidad de pasos determinada.
    """
    reachable_states = env.get_reachable_states(start_x, start_y, num_pasos)
    return np.log2(len(reachable_states))

def calcular_empowerment_tablero(env, num_pasos):
    """
    Calcula el empowerment de cada celda en el tablero y lo almacena en una matriz.
    """
    empowerment_grid = np.zeros((env.grid_size, env.grid_size))
    
    for x in range(1, env.grid_size - 1):
        for y in range(1, env.grid_size - 1):
            empowerment = calcular_empowerment(env, x, y, num_pasos)
            empowerment_grid[x, y] = empowerment
    
    return empowerment_grid

env = MiniGridEnv()
num_pasos = 22  
empowerment_grid = calcular_empowerment_tablero(env, num_pasos)

# Heatmap
plt.figure(figsize=(8, 8))
plt.imshow(empowerment_grid, cmap="viridis", origin="upper")
plt.colorbar(label="Empowerment")
plt.title(f"Heatmap de Empowerment en el Tablero a {num_pasos} Pasos")
plt.xlabel("Posición X")
plt.ylabel("Posición Y")
plt.show()
