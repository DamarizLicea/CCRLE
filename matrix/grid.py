import numpy as np
import matplotlib.pyplot as plt

# Movimientos posibles
movements = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1)
}

def is_valid_position(pos, grid_size):
    return 0 <= pos[0] < grid_size and 0 <= pos[1] < grid_size

def simulate_agent(start_pos, grid_size, steps):
    visited = set()  # Celdas visitadas, con set() evita duplicados
    current_positions = {start_pos}  # Posiciones iniciales
    
    for step in range(steps):
        # Nuevas posiciones alcanzadas en este paso, el set() evita duplicados
        new_positions = set()
        for pos in current_positions:
            visited.add(pos)  # Añadir la celda a las alcanzadas
            # Hacer cada movimiento posible
            for move in movements.values():
                new_pos = (pos[0] + move[0], pos[1] + move[1])
                if is_valid_position(new_pos, grid_size):
                    new_positions.add(new_pos)
        current_positions = new_positions
    visited.update(current_positions)
    
    return visited

# n step empowerment
def calculate_empowerment_n_steps(start_pos, grid_size, steps):
    reached_states = simulate_agent(start_pos, grid_size, steps)
    num_states_reached = len(reached_states)
    if num_states_reached > 0:
        empowerment = np.log2(num_states_reached)
    else:
        empowerment = 0

    return empowerment

# heatmap
def generate_empowerment_heatmap(grid_size, steps):
    empowerment_matrix = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            start_position = (i, j)
            empowerment_matrix[i, j] = calculate_empowerment_n_steps(start_position, grid_size, steps)
    
    plt.imshow(empowerment_matrix, cmap='viridis', origin='upper')
    plt.colorbar(label='Empowerment')
    plt.title(f'Heatmap de Empowerment ({steps} pasos)')
    plt.show()

# tamaño de la cuadrícula
grid_size = 5  

# pasos para calcular el empowerment
n_steps = 4

generate_empowerment_heatmap(grid_size, n_steps)