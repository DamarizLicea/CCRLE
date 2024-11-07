from __future__ import annotations
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall
from minigrid.minigrid_env import MiniGridEnv
import numpy as np
import random
import time

class PassableWall(Wall):
    """ Subclase de Wall que permite que el agente pueda pasar sobre ella. 
        Tambien se le puede decir pasillo"""
    def can_overlap(self):
        return True


class SimpleEnv(MiniGridEnv):
    """ Clase que importa MiniGridEnv (EmptyEnv)
        para generar el entorno."""
    def __init__(self, size=13, agent_start_pos=(6, 6), agent_start_dir=0, max_steps: int | None = None, **kwargs):
        """ Constructor de la clase que setea los parametros
            para un entorno de 13 x 13 con un agente en la posición (6, 6) y
            donde la dirección del agente se bloquea a 0"""
        self.agent_start_pos = agent_start_pos 
        self.agent_start_dir = agent_start_dir
        mission_space = MissionSpace(mission_func=self._gen_mission)
        self.quadrants = [(3, 3), (3, 9), (9, 3), (9, 9)]  
        self.current_quadrant = None
        self.reward_positions = []  

        if max_steps is None:
            max_steps = 4 * size**2
        
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )
        self.empowerment_matrix = np.full((size, size), -1)

    @staticmethod
    def _gen_mission():
        return "Obtener todas las recompensas."

    def _gen_grid(self, width, height):
        """ Crea el grid del entorno segun los parametros especificados, 
            añade pasillos y coloca las recompensas en un cuadrante aleatorio.
            Además se calcula el empowerment para cada celda del grid."""

        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Colocar pasillos
        for i in range(1, height - 1):
            self.grid.set(6, i, PassableWall())
        for i in range(1, width - 1):
            self.grid.set(i, 6, PassableWall())

        self.current_quadrant = random.choice(self.quadrants)
        self.place_rewards_in_quadrant()

        self.calculate_empowerment_matrix()
        max_empowerment_pos = self.find_max_empowerment_position()
        self.agent_pos = self.agent_start_pos
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

    def calculate_empowerment_matrix(self, n_steps=10):
        """ Función que utiliza calculate_empowerment para llenar una 
            matriz para todas las celdas accesibles en el entorno."""
        for row in range(1, self.grid.height - 1):
            for col in range(1, self.grid.width - 1):
                cell = self.grid.get(col, row)
                # Solo omitir celdas de Wall
                if not isinstance(cell, Wall) or isinstance(cell, PassableWall):
                    self.empowerment_matrix[row, col] = self.calculate_empowerment(col, row, n_steps)


    def calculate_empowerment(self, x, y, n_steps=2):
        """ Función que calcula el empowerment de una celda dada, al calcular
            el logaritmo del número de celdas alcanzables con n pasos."""
        reachable_states = set()

        def explore_positions(pos, steps):
            if steps == 0:
                return
            for direction in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # Movimientos: derecha, abajo, izquierda, arriba
                new_x, new_y = pos[0] + direction[0], pos[1] + direction[1]

                if 0 <= new_x < self.grid.width and 0 <= new_y < self.grid.height:
                    cell = self.grid.get(new_x, new_y)
                    if not isinstance(cell, Wall) or isinstance(cell, PassableWall):
                        if (new_x, new_y) not in reachable_states:
                            reachable_states.add((new_x, new_y))
                            explore_positions((new_x, new_y), steps - 1)
        
        explore_positions((x, y), n_steps)
        
        # Considerar recompensas
        if (x, y) in self.reward_positions:
            empowerment_value = np.log2(len(reachable_states)) + 2 # Es recompensa, bonus.
        else:
            empowerment_value = np.log2(len(reachable_states))
        
        return empowerment_value



    def find_max_empowerment_position(self):
        """ Función que encuentra la posición con el máximo valor de empowerment en el tablero. """
        max_empowerment = -1
        max_pos = (self.agent_start_pos) 

        for row in range(1, self.grid.height - 1):
            for col in range(1, self.grid.width - 1):
                empowerment_value = self.empowerment_matrix[row, col]
                if empowerment_value > max_empowerment:
                    max_empowerment = empowerment_value
                    max_pos = (col, row)
        
        print(f"Posición de máximo empowerment: {max_pos} con un valor de {max_empowerment}")
        return max_pos

    def save_empowerment_matrix(self, filename="empowerment_matrix.txt"):
        """ Función que guarda la matriz de empowerment en un archivo de texto """
        np.savetxt(filename, self.empowerment_matrix, fmt="%.2f")

    def encode_state(self, x, y):
        """ Función para codificar el estado dada la posición del agente y el cuadrante de las recompensas."""
        quadrant_index = self.quadrants.index(self.current_quadrant)
        return (x * 100 + y) + quadrant_index 

    def auto_move(self):
        """ Función para mover al agente automáticamente por todo el tablero
            al calcular el empowerment. """
        for row in range(1, self.grid.height - 1):
            for col in range(1, self.grid.width - 1):
                self.agent_pos = (col, row)
                self.agent_dir = 0
                self.render()
                print(f"Agente en posición (x, y): ({col}, {row}) | Estado codificado: {self.encode_state(col, row)}")
                time.sleep(0.1)

        center_x, center_y = self.grid.width // 2, self.grid.height // 2
        self.agent_pos = (center_x, center_y)
        self.agent_dir = 0
        self.render()

    # ------------------ Q-learning ------------------ #
    def initialize_q_table(self):
        """ Función para inicializar la Q-table."""
        self.q_table = {}
        for row in range(1, self.grid.height - 1):
            for col in range(1, self.grid.width - 1):
                state = self.encode_state(col, row)
                self.q_table[state] = {0: 0, 1: 0, 2: 0, 3: 0}

    def save_q_table(self, filename="q_table.txt"):
        """ Función para guardar la Q-table en un archivo de texto."""
        with open(filename, "w") as file:
            for state, actions in self.q_table.items():
                file.write(f"State {state}: {actions}\n")

    def q_learning_agent(self, alpha=0.5, gamma=0.9, epsilon=0.9, min_epsilon=0.01, decay_rate=0.99, max_steps=90, episodes=7000):
            """ Función para representar al segundo agente, que usa reinforcement learning
                mediante Q-Learning, el objetivo de este agente es recolectar las recompensas
                en el menor numero de pasos posibles. """
            
            min_steps_to_goal = float('inf')
            best_episode = None
            successful_episodes = 0

            for episode in range(episodes):
                current_pos = self.agent_start_pos
                steps = 0
                rewards_collected = 0
                total_reward = 0 

                self._gen_grid(self.grid.width, self.grid.height)
                print(f"\nIniciando episodio {episode + 1}")

                while steps < max_steps:
                    state = self.encode_state(*current_pos)

                    if state not in self.q_table:
                        self.q_table[state] = {0: 0, 1: 0, 2: 0, 3: 0}

                    # Exploración o explotación
                    if random.uniform(0, 1) < epsilon:
                        action = random.choice(list(self.q_table[state].keys()))  # Exploración
                    else:
                        action = max(self.q_table[state], key=self.q_table[state].get)  # Explotación

                    new_pos = self.next_position(current_pos, action)
                    new_state = self.encode_state(*new_pos)

                    if new_state not in self.q_table:
                        self.q_table[new_state] = {0: 0, 1: 0, 2: 0, 3: 0}

                    reward = -1

                    if new_pos in self.reward_positions:
                        reward += 1 
                        rewards_collected += 1
                        self.reward_positions.remove(new_pos)
                        self.grid.set(*new_pos, None)
                        print(f"Recompensa encontrada en posición {new_pos}. Total: {rewards_collected}")
                        self.render()

                        if rewards_collected == 4:
                            reward += 10
                            successful_episodes += 1 
                            total_reward += reward
                            print(f"Episodio {episode + 1} completado en {steps + 1} pasos con una recompensa total de {total_reward}")

                            if steps + 1 < min_steps_to_goal:
                                min_steps_to_goal = steps + 1
                                best_episode = episode + 1
                            break

                    total_reward += reward

                    self.q_table[state][action] += alpha * (reward + gamma * max(self.q_table[new_state].values()) - self.q_table[state][action])
                    current_pos = new_pos
                    self.agent_pos = current_pos
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
            if not isinstance(self.grid.get(new_x, new_y), Wall) or isinstance(self.grid.get(new_x, new_y), PassableWall):
                return new_x, new_y
        
        return pos

    def run_agents(self):
        """Función auxiliar para ejecutar los agentes en secuencia."""
        self.reset()
        print("Ejecutando el primer agente (empowerment).")
        self.auto_move()
        print("El primer agente terminó su recorrido.\nIniciando al agente de Q-Learning...")
        
        self.initialize_q_table()
        self.q_learning_agent()

def main():
    env = SimpleEnv(render_mode="human")

    env.run_agents()
    env.calculate_empowerment_matrix()
    env.save_empowerment_matrix()
    env.close()

if __name__ == "__main__":
    main()
