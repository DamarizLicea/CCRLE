
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
    def __init__(self, size=13, agent_start_pos=(2, 2), agent_start_dir=0, num_pasos = 6, max_steps: int | None = None, **kwargs):
        """ Constructor de la clase que setea los parametros
            para un entorno de 13 x 13 con un agente en la posición (6, 6) y
            donde la dirección del agente se bloquea a 0"""
        self.agent_start_pos = agent_start_pos 
        self.agent_start_dir = agent_start_dir
        self.num_pasos = num_pasos
        mission_space = MissionSpace(mission_func=self._gen_mission)
        self.quadrants = [(3, 3), (3, 9), (9, 3), (9, 9)]  
        self.current_quadrant = None
        self.reward_positions = []
        self.grid_size = 13  

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
        return (x * 100 + y) + quadrant_index 

    def auto_move(self):
        """ Función para mover al agente automáticamente por todo el tablero
            al calcular el empowerment. """
        for row in range(1, self.grid.height - 1):
            for col in range(1, self.grid.width - 1):
                self.agent_pos = (col, row)
                self.agent_dir = 0
                self.render()
                print(f"Agente en posición (x, y): ({col}, {row})")
                time.sleep(0.1)

        center_x, center_y = self.grid.width // 2, self.grid.height // 2
        self.agent_pos = (center_x, center_y)
        self.agent_dir = 0
        self.render()

    # ------------------ Q-learning ------------------ #
                    # acciones: 0 = abajo, 1 = derecha, 2 = arriba, 3 = izquierda
                    # 3 puede ser izquierda
                    # 0 puede ser abajo
                    # 2 puede ser arriba
                    # 1 puede ser derecha
                    # desde la posicion (5,6) (y,x) se puede ir a (5,5), (6,6), (5,7), (4,6)


    def save_q_table(self, filename="q_table_empql2.txt"):
        """ Función para guardar la Q-table en un archivo de texto."""
        with open(filename, "w") as file:
            for state, actions in self.q_table.items():
                file.write(f"State {state}: {actions}\n")

    def get_state_value(self, state):
        """Función auxiliar para obtener o crear valores para un estado."""
        if state not in self.q_table:
            self.q_table[state] = {}
            # Decode posición x,y del estado
            x = (state - (state % 4)) // 100
            y = (state - (state % 4)) % 100
        
            for action in range(4):
                next_pos = self.next_position((x, y), action)
                if 0 <= next_pos[0] < self.grid.width and 0 <= next_pos[1] < self.grid.height:
                    emp_value = self.empowerment_grid[next_pos[1], next_pos[0]]
                else:
                    emp_value = 0
                self.q_table[state][action] = random.uniform(0, 0.1)
        return self.q_table[state]

    def initialize_q_table(self):
        """Inicialización de la Q-table con valores aleatorios pequeños."""
        self.q_table = {}
        
        # Para cada posición válida en el grid
        for row in range(self.grid.height):
            for col in range(self.grid.width):
                for quadrant_index in range(len(self.quadrants)):
                    state = (col * 100 + row) + quadrant_index
                    self.q_table[state] = {
                        action: random.uniform(0, 0.1) for action in range(4)
                    }

    def next_position(self, pos, action):
        """Función que calcula la siguiente posición basada en la acción."""
        direction = [(0, 1), (1, 0), (0, -1), (-1, 0)][action]
        new_x, new_y = pos[0] + direction[0], pos[1] + direction[1]
        
        # Verificar límites y paredes
        if (0 <= new_x < self.grid.width and 
            0 <= new_y < self.grid.height and 
            not isinstance(self.grid.get(new_x, new_y), Wall)):
            return new_x, new_y
        return pos

    def q_learning_agent(self, alpha=0.1, gamma=0.9, epsilon=1.0, min_epsilon=0.01, decay_rate=0.995, max_steps=70, episodes=1500):
        """Q-Learning modificado para mejor exploración y aprendizaje."""
        min_steps_to_goal = float('inf')
        best_episode = None
        successful_episodes = 0
        episode_rewards = []
        
        # Inicializar la Q-table
        self.initialize_q_table()
        
        for episode in range(episodes):
            current_pos = self.agent_start_pos
            steps = 0
            rewards_collected = 0
            episode_reward = 0
            
            self._gen_grid(self.grid.width, self.grid.height)
            current_quadrant = self.quadrants.index(self.current_quadrant)
            print(f"\nIniciando episodio {episode + 1}")
            
            while steps < max_steps:
                state = self.encode_state(*current_pos) + current_quadrant
                
                # Exploración vs. Explotación
                if random.uniform(0, 1) < epsilon:
                    action = random.randint(0, 3)
                else:
                    state_values = self.get_state_value(state)
                    action = max(state_values, key=state_values.get)
                
                new_pos = self.next_position(current_pos, action)
                new_state = self.encode_state(*new_pos) + current_quadrant
                
                reward = -0.1  # Penalización por paso
                
                # Bonus de empowerment si la posición es válida
                if 0 <= new_pos[1] < self.grid.height and 0 <= new_pos[0] < self.grid.width:
                    emp_reward = self.empowerment_grid[new_pos[1], new_pos[0]] 
                
                # Recompensa por encontrar objetivo
                if new_pos in self.reward_positions:
                    reward += 10
                    rewards_collected += 1
                    self.reward_positions.remove(new_pos)
                    self.grid.set(*new_pos, None)
                    print(f"Recompensa encontrada en posición {new_pos}. Total: {rewards_collected}")
                
                # Recompensa por completar la tarea
                if rewards_collected == 4:
                    reward += 50
                    successful_episodes += 1
                    print(f"Episodio {episode + 1} completado en {steps + 1} pasos")
                    if steps + 1 < min_steps_to_goal:
                        min_steps_to_goal = steps + 1
                        best_episode = episode + 1
                    break
                
                # Actualización Q-learning
                next_state_values = self.get_state_value(new_state)
                max_next_q = max(next_state_values.values())
                current_q = self.q_table[state].get(action, 0)
                
                # Fórmula Q-learning
                new_q = current_q + alpha * (emp_reward + gamma * max_next_q - current_q)
                self.q_table[state][action] = new_q

                current_pos = new_pos
                self.agent_pos = current_pos
                episode_reward += reward
                
                self.render()
                time.sleep(0.1)
                steps += 1
                
                if steps >= max_steps:
                    print(f"Límite de pasos alcanzado en episodio {episode + 1}")
            
            epsilon = max(min_epsilon, epsilon * decay_rate)
            episode_rewards.append(episode_reward)
            
            print(f"Episodio {episode + 1} terminado. Recompensa total: {episode_reward:.2f}")
        
        print(f"\nEntrenamiento completado con {successful_episodes} episodios exitosos de {episodes}")
        if best_episode is not None:
            print(f"Mejor episodio: {best_episode} con {min_steps_to_goal} pasos")
        else:
            print("No se lograron recoger las 4 recompensas en ningún episodio")
        
        self.save_q_table()
        return episode_rewards

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
        print("Ejecutando el primer agente (empowerment).")
        self.auto_move()
        print("El primer agente terminó su recorrido.\nIniciando al agente de Q-Learning...")
        
        self.initialize_q_table()
        self.q_learning_agent()

def main():
    num_pasos = 6 
    env = SimpleEnv(render_mode="human", num_pasos=num_pasos)
    env.run_agents()
    empowerment_grid = env.calculate_empowerment_matrix()
    env.save_empowerment_matrix()

    env.close()

if __name__ == "__main__":
    main()

