from taxiTwo import calculate_empowerment

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

def calculate_empowerment_grid(env):
    grid_empowerment = np.zeros((5, 5))
    num_combinations = 5 * 4  # 5 del pasajero * 4 destinos
    
    for row in range(5):
        for col in range(5):
            empowerment_sum = 0.0
            for passenger in range(5):
                for destination in range(4):
                    state = env.unwrapped.encode(row, col, passenger, destination)
                    empowerment = calculate_empowerment(env, state, epsilon=1e-10)
                    empowerment_sum += empowerment

            # Calcular el promedio de empowerment para la celda (row, col)
            grid_empowerment[row, col] = empowerment_sum / num_combinations

    return grid_empowerment

def plot_empowerment_heatmap(empowerment_grid):
    print("Empowerment Grid:", empowerment_grid)
    min_empowerment = empowerment_grid.min()
    max_empowerment = empowerment_grid.max()
    print(f"Min empowerment: {min_empowerment}, Max empowerment: {max_empowerment}")
    
    plt.figure(figsize=(6, 6))
    plt.imshow(empowerment_grid, cmap="YlGnBu", origin="upper", vmin=min_empowerment, vmax=max_empowerment)
    plt.colorbar(label="Empowerment")
    plt.title("Heatmap de Empowerment Entorno Taxi")
    plt.xlabel("Columna del Entorno")
    plt.ylabel("Fila del Entorno")
    plt.show()

env = gym.make('Taxi-v3')
empowerment_grid = calculate_empowerment_grid(env)
plot_empowerment_heatmap(empowerment_grid)