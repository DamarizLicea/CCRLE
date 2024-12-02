from __future__ import annotations
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall
from minigrid.minigrid_env import MiniGridEnv
import matplotlib.pyplot as plt
import numpy as np
import random
import time


class SimpleEnv(MiniGridEnv):
    """ Clase que importa MiniGridEnv (EmptyEnv)
        para generar el entorno."""
    def __init__(self, size=13, agent_start_pos=(6, 6), agent_start_dir=0, num_pasos = 8 , max_steps: int | None = None, **kwargs):
        """ Constructor de la clase que setea los parametros
            para un entorno de 13 x 13 con un agente en la posición (6, 6) y
            donde la dirección del agente se bloquea a 0"""
        
        self.agent_start_dir = agent_start_dir
        self.num_pasos = num_pasos
        mission_space = MissionSpace(mission_func=self._gen_mission)
        self.quadrants = [(3, 3), (3, 9), (9, 3), (9, 9)]  
        self.current_quadrant = None
        self.reward_positions = []
        self.grid_size = 13
        self.agent_start_pos = (random.randint(1, self.grid_size - 2), random.randint(1, self.grid_size- 2))

        if max_steps is None:
            max_steps = 4 * size**2
        
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )
        self.empowerment_grid = np.full((size, size), -1)


    @staticmethod
    def _gen_mission():
        return "Obtener todas las recompensas."

    def _gen_grid(self, width, height):
        """ Crea el grid del entorno segun los parametros especificados, 
            añade pasillos y coloca las recompensas en un cuadrante aleatorio.
            Además se calcula el empowerment para cada celda del grid."""

        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        self.current_quadrant = random.choice(self.quadrants)
        self.place_rewards_in_quadrant()
        self.empowerment_grid = self.calculate_empowerment_matrix()
        max_empowerment_pos = self.find_max_empowerment_position()
        self.agent_pos = (random.randint(1, self.grid_size - 2), random.randint(1, self.grid_size - 2))
        self.agent_dir = self.agent_start_dir
        self.mission = "Obtener todas las recompensas."
        self.render()
        time.sleep(2) 

    def place_rewards_in_quadrant(self):
        """ Funcion que coloca las recompensas en un cuadrante aleatorio. """
        x_center, y_center = self.current_quadrant

        self.reward_positions = [
            (x_center - 1, y_center - 1), (x_center - 1, y_center + 1),
            (x_center + 1, y_center - 1), (x_center + 1, y_center + 1)
        ]

        for pos in self.reward_positions:
            self.put_obj(Goal(), *pos)
    
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
                if 1 <= nx <= 11 and 1 <= ny <= 11:  # Asumiendo límites del tablero
                    dfs(nx, ny, remaining_steps - 1)
        
        dfs(start_x, start_y, steps)
        return reachable_states

    def calculate_empowerment(self, start_x, start_y):
        """
        Calcula el empowerment en función de los estados únicos alcanzables desde la posición inicial
        en una cantidad de pasos determinada.
        """
        reachable_states = self.get_reachable_states(start_x, start_y, self.num_pasos)
        return np.log2(len(reachable_states)) if reachable_states else 0

    def calculate_empowerment_matrix(self):
        """
        Calcula el empowerment de cada celda en el tablero y lo almacena en una matriz.
        """
        empowerment_grid = np.zeros((self.grid_size, self.grid_size))
        
        for x in range(1, self.grid_size - 1):  # Evitar bordes de paredes
            for y in range(1, self.grid_size - 1):
                empowerment = self.calculate_empowerment(x, y)
                empowerment_grid[x, y] = empowerment
        
        return empowerment_grid

    def find_max_empowerment_position(self):
            """ Función que encuentra la posición con el máximo valor de empowerment en el tablero. """
            max_empowerment = -1
            max_pos = (self.agent_start_pos) 

            for row in range(1, self.grid.height - 1):
                for col in range(1, self.grid.width - 1):
                    empowerment_value = self.empowerment_grid[row, col]
                    if empowerment_value > max_empowerment:
                        max_empowerment = empowerment_value
                        max_pos = (col, row)

            print(f"Posición de máximo empowerment: {max_pos} con un valor de {max_empowerment}")
            return max_pos


    def save_empowerment_matrix(self, filename="empowerment_matrix.txt"):
        """ Función que guarda la matriz de empowerment en un archivo de texto """

        np.savetxt(filename, self.empowerment_grid, fmt="%.2f")


    def encode_state(self, x, y):
        """ Función para codificar el estado dada la posición del agente y el cuadrante de las recompensas."""
        quadrant_index = self.quadrants.index(self.current_quadrant)
        # quadrant_index = 
        return (x * 100 + y * 10) + quadrant_index 


    # ------------------ Q-learning ------------------ #
    def initialize_q_table(self):
        """ Función para inicializar la Q-table con todas las combinaciones posibles de estado."""
        self.q_table = {}
        for row in range(1, self.grid.height - 1):
            for col in range(1, self.grid.width - 1):
                for quadrant_index in range(len(self.quadrants)):
                    # Codificamos el estado con el cuadrante de recompensa
                    state = (col * 100 + row * 10) + quadrant_index
                    self.q_table[state] = {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1}

    def save_q_table(self, filename="q_tableA.txt"):
        """ Función para guardar la Q-table en un archivo de texto."""
        with open(filename, "w") as file:
            for state, actions in self.q_table.items():
                file.write(f"State {state}: {actions}\n")

    def q_learning_agent(self, alpha=0.2, gamma=0.9, epsilon=0.92, min_epsilon=0.01, decay_rate=0.98, max_steps=90, episodes=1500):
            """ Función para representar al segundo agente, que usa reinforcement learning
                mediante Q-Learning, el objetivo de este agente es recolectar las recompensas
                en el menor numero de pasos posibles. """
            
            min_steps_to_goal = float('inf')
            best_episode = None
            successful_episodes = 0
            visited_states = set()

            for episode in range(episodes):
                current_pos = self.agent_start_pos
                steps = 0
                rewards_collected = 0
                total_reward = 0 

                self._gen_grid(self.grid.width, self.grid.height)
                current_quadrant= self.quadrants.index(self.current_quadrant)
                print(f"\nIniciando episodio {episode + 1}")

                current_reward_positions = self.reward_positions.copy()

                while steps < max_steps:
                    state = self.encode_state(*current_pos) + current_quadrant
                    # Asegurar que el estado existe en la Q-table
                    if state not in self.q_table:
                        self.q_table[state] = {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1}

                    # Exploración o explotación con probabilidad epsilon
                    if random.uniform(0, 1) < epsilon:
                        action = random.choice(list(self.q_table[state].keys())) # Exploración
                    else:
                        action = max(self.q_table[state], key=self.q_table[state].get)  # Explotación
                    # Calcular nueva posición
                    new_pos = self.next_position(current_pos, action)
                    new_state = self.encode_state(*new_pos) + current_quadrant

                    # Inicializar nuevo estado si no existe
                    if new_state not in self.q_table:
                        self.q_table[new_state] = {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1}

                    reward = -0.01
                    if new_pos in current_reward_positions:
                        reward += 40
                        rewards_collected += 1
                        print(f"Recompensa encontrada en {new_pos}. Recompensas recogidas: {rewards_collected}")
                        current_reward_positions.remove(new_pos)
                        self.grid.set(*new_pos, None)
                        self.reward_positions.remove(new_pos)
                        visited_states.add(current_pos)
                        if current_pos in visited_states:
                            reward -= 10

                        # como lo multo si se cicla en una celda donde ya no hay recompensa?

                        if rewards_collected == 4:
                            reward += 100
                            successful_episodes += 1
                            print(f"¡Todas las recompensas recogidas en {steps + 1} pasos!")
                            if steps+1 < min_steps_to_goal:
                                min_steps_to_goal = steps+1
                                best_episode = episode + 1
                            break

                    total_reward += reward
                    
                    self.q_table[state][action] += alpha * (reward + gamma * max(self.q_table[new_state].values()) - self.q_table[state][action])

                    current_pos = new_pos
                    self.agent_pos = current_pos

                    # Decaimiento de epsilon
                    epsilon = max(min_epsilon, epsilon * decay_rate)

                    self.render()
                    steps += 1

                    if steps >= max_steps:
                        print(f"Límite de pasos alcanzado en el episodio {episode + 1}. Recompensa total: {total_reward}")
                        break

            # Outputs de resumen de episodios
            print(f"\nEntrenamiento completado con {successful_episodes} episodios exitosos de {episodes}")
            if best_episode is not None:
                print(f"Mejor episodio: {best_episode} con {min_steps_to_goal} pasos.")
            else:
                print("No se lograron recoger las 4 recompensas en ningún episodio.")
            self.save_q_table()

    def next_position(self, pos, action):
        """Función que calcula la siguiente posición basada en la acción."""
        direction = [(0, 1), (1, 0), (0, -1), (-1, 0)][action]
        new_x, new_y = pos[0] + direction[0], pos[1] + direction[1]
        
        if 0 <= new_x < self.grid.width and 0 <= new_y < self.grid.height:
            if not isinstance(self.grid.get(new_x, new_y), Wall):
                return new_x, new_y
        
        return pos

    def run_agents(self):
        """Función auxiliar para ejecutar los agentes en secuencia."""
        self.reset()
        print("El primer agente terminó su recorrido.\nIniciando al agente de Q-Learning...")
        
        self.initialize_q_table()
        self.q_learning_agent()

def main():
    num_pasos = 6 
    env = SimpleEnv(render_mode="human", num_pasos=num_pasos)

    env.run_agents() # Número de pasos para calcular el empowerment
    empowerment_grid = env.calculate_empowerment_matrix()
    env.save_empowerment_matrix()

    plt.figure(figsize=(8, 8))
    plt.imshow(empowerment_grid, cmap="viridis", origin="upper")
    plt.colorbar(label="Empowerment")
    plt.title(f"Heatmap de Empowerment en el Tablero a {num_pasos} Pasos")
    plt.xlabel("Posición X")
    plt.ylabel("Posición Y")
    plt.show()
    env.close()

if __name__ == "__main__":
    main()
