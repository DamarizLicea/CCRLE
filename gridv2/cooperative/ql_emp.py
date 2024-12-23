from __future__ import annotations
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.constants import COLOR_NAMES, COLOR_TO_IDX, IDX_TO_COLOR
from minigrid.core.world_object import Goal, Wall
from minigrid.minigrid_env import MiniGridEnv
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import os

"""
Este script implementa un entorno de MiniGrid que combina los agentes Empowerment y RL.
El agente Empowerment se entrena con Q-learning usando el empowerment como recompensa.
El agente RL se entrena con Q-learning normal y epsilon greedy.
El entorno se ejecuta en episodios, alternando entre los dos agentes en cada episodio.
"""


class CombinedEnv(MiniGridEnv):
    def __init__(self, size=13, emp_agent_start_pos=(2, 2), rl_agent_start_pos=(6, 6), 
                 emp_agent_start_dir=0, rl_agent_start_dir=0, 
                 emp_agent_num_pasos=6, rl_agent_num_pasos=6, 
                 max_steps: int | None = None, **kwargs):
        
        self.emp_agent_start_pos = emp_agent_start_pos
        self.rl_agent_start_pos = rl_agent_start_pos
        self.emp_agent_start_dir = emp_agent_start_dir
        self.rl_agent_start_dir = rl_agent_start_dir
        self.emp_agent_num_pasos = emp_agent_num_pasos
        self.rl_agent_num_pasos = rl_agent_num_pasos
        
        mission_space = MissionSpace(mission_func=self._gen_mission)
        self.quadrants = [(3, 3), (3, 9), (9, 3), (9, 9)]  
        self.current_quadrant = None
        self.reward_positions = []
        self.grid_size = 13
        self.total_episodes = 20
        self.remaining_rewards = 1
        self.collected_rewards = 0
        self.successEP = False
        self.successE = False
        self.current_agent = 'rl'

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
        self.reward_range = (0, 1)
        
        # Inicializar Q-tables separadas
        self.q_table_emp_agent = {}
        self.q_table_rl_agent = {}

        self.current_agent_pos = self.emp_agent_start_pos
        self.current_agent_dir = self.emp_agent_start_dir

    @staticmethod
    def _gen_mission():
        return "Obtener todas las recompensas."

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        self.current_quadrant = random.choice(self.quadrants)
        self.place_rewards_in_quadrant()
        self.empowerment_grid = self.calculate_empowerment_matrix()
        self.agent_pos = self.current_agent_pos
        self.agent_dir = self.current_agent_dir
        self.mission = "Obtener todas las recompensas."
        self.render()
        time.sleep(2)

    def place_rewards_in_quadrant(self):
        """ Coloca una recompensa en el centro del cuadrante actual. """
        x_center, y_center = self.current_quadrant
        reward_color = 'grey' if self.current_agent == 'emp' else 'green'
        self.reward_positions = [(x_center, y_center)]
        rewar = Goal()
        rewar.color = reward_color
        self.put_obj(rewar, x_center, y_center)


    def get_reachable_states(self, start_x, start_y, steps=1):
        """ Obtiene los estados alcanzables desde una posición inicial en un número de pasos dado. """

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

    def calculate_empowerment(self, start_x, start_y, num_pasos):
        """ Calcula el empowerment de una posición inicial en un número de pasos dado. """
        reachable_states = self.get_reachable_states(start_x, start_y, num_pasos)
        return np.log2(len(reachable_states)) if reachable_states else 0

    def calculate_empowerment_matrix(self):
        """ Calcula el empowerment para cada celda del grid y lo almacena en una matriz. """
        empowerment_grid = np.zeros((self.grid_size, self.grid_size))
        
        for x in range(1, self.grid_size - 1):
            for y in range(1, self.grid_size - 1):
                empowerment = self.calculate_empowerment(x, y, self.emp_agent_num_pasos)
                empowerment_grid[x, y] = empowerment
        
        return empowerment_grid

    def encode_state(self, x, y):
        """ Codifica un estado en un número entero único. """
        quadrant_index = self.quadrants.index(self.current_quadrant)
        return (x * 100 + y) + quadrant_index

    def initialize_q_table(self, agent_num):
        """ Inicializa la Q-table para un agente dado. """
        q_table = {}
        for row in range(1, self.grid.height - 1):
            for col in range(1, self.grid.width - 1):
                for quadrant_index in range(len(self.quadrants)):
                    state = (col * 100 + row) + quadrant_index
                    q_table[state] = {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1}
        
        if agent_num == 1:
            self.q_table_emp_agent = q_table
        else:
            self.q_table_rl_agent = q_table

    def load_q_table(self, filename):
        q_table = {}
        if os.path.exists(filename):
            with open(filename, "r") as file:
                for line in file:
                    parts = line.strip().split(": ")
                    if len(parts) == 2:
                        state, actions = parts
                        state = int(state.split()[1])
                        actions = eval(actions)
                        q_table[state] = actions
        return q_table

    def save_q_table(self, q_table, filename):
        with open(filename, "w") as file:
            for state, actions in q_table.items():
                file.write(f"State {state}: {actions}\n")

    def next_position(self, pos, action):
        """ Obtiene la siguiente posición en base a una acción dada. """
        direction = [(0, 1), (1, 0), (0, -1), (-1, 0)][action]
        new_x, new_y = pos[0] + direction[0], pos[1] + direction[1]
        
        if 0 <= new_x < self.grid.width and 0 <= new_y < self.grid.height:
            if not isinstance(self.grid.get(new_x, new_y), Wall):
                return new_x, new_y
        
        return pos
    
    def get_state_value(self, state, agent='emp'):
        """Obtiene el valor del estado de la Q-table del agente especificado."""
        if agent == 'emp':
            q_table = self.q_table_emp_agent
        elif agent == 'rl':
            q_table = self.q_table_rl_agent
        else:
            raise ValueError("El agente especificado debe ser 'emp' o 'rl'.")
        
        if state not in q_table:
            q_table[state] = {action: random.uniform(0, 0.1) for action in range(4)}
        
        return q_table[state]
    
    def reset_environment(self):
        """Resetea el ambiente para prepararlo para un nuevo turno"""
        self._gen_grid(self.grid.width, self.grid.height)
        self.current_quadrant = random.choice(self.quadrants)
        self.place_rewards_in_quadrant()
        self.empowerment_grid = self.calculate_empowerment_matrix()


    def q_emp_agent(self, alpha=0.1, gamma=0.9, epsilon=1.0, min_epsilon=0.01, decay_rate=0.995, max_steps=120, episodes=5):
        """Entrena a un agente usando q-learning con empowerment como recompensa."""
        max_emp_pos = None
        successE = False
        max_emp = np.max(self.empowerment_grid)
        self.current_agent = 'emp'
        max_emp_pos = np.where(self.empowerment_grid == max_emp)
        max_emp_pos = (max_emp_pos[0][0], max_emp_pos[1][0])
        print(f"Posición de máximo empowerment: {max_emp_pos}")
        print(f"Empowerment máximo: {max_emp}")
        self.q_table_emp_agent = self.load_q_table("q_table_empql21.txt")
        if not self.q_table_emp_agent:
            self.initialize_q_table(1)
        for episode in range(episodes):
            current_pos = self.emp_agent_start_pos
            steps = 0
            self._gen_grid(self.grid.width, self.grid.height)
            current_quadrant = self.quadrants.index(self.current_quadrant)
            print(f"Recompensas en {self.reward_positions}")
            print(f"Recompensas restantes: {self.remaining_rewards}")
            while steps < max_steps:
                state = self.encode_state(*current_pos) + current_quadrant
                    
                if state not in self.q_table_emp_agent:
                    self.q_table_emp_agent[state] = {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1}
                    # Epsilon greedy
                if random.uniform(0, 1) < epsilon:
                        action = random.randint(0, 3)
                else:
                        action = max(self.q_table_emp_agent[state], key=self.q_table_emp_agent[state].get)
                    
                new_pos = self.next_position(current_pos, action)
                new_state = self.encode_state(*new_pos) + current_quadrant
                if new_state not in self.q_table_emp_agent:
                        self.q_table_emp_agent[new_state] = {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1}
                emp_reward = float(self.empowerment_grid[new_pos[1], new_pos[0]])
                #print(f"Empowerment en {new_pos}: {emp_reward}")
                    # Actualización Q-learning, picadita
                self.q_table_emp_agent[state][action] += alpha * (emp_reward + gamma * max(self.q_table_emp_agent[new_state].values()) - self.q_table_emp_agent[state][action])
                current_pos = new_pos
                self.agent_pos = current_pos
                
                self.render()
                time.sleep(0.1)
                steps += 1
                epsilon = max(min_epsilon, epsilon * decay_rate)

                if new_pos == max_emp_pos:
                        print(f"Agente emp en posición de máximo empowerment.")
                        successE = True
                        break
                    
                if steps >= max_steps:
                        break
                
        self.save_q_table(self.q_table_emp_agent, "q_table_empql21.txt")
            
        return successE

    def q_rl_agent(self, alpha=0.2, gamma=0.9, epsilon=0.9, min_epsilon=0.01, decay_rate=0.99, max_steps=70, episodes=5):
        """Entrena a un agente usando q-learning normal y epsilon greedy ."""
        min_steps_to_goal = float('inf')
        best_episode = None
        successful_episodes = 0
        self.current_agent= 'rl'
        self.q_table_rl_agent = self.load_q_table("q_table21.txt")
        if not self.q_table_rl_agent:
            self.initialize_q_table(2)
        
        for episode in range(episodes):
            current_pos = self.rl_agent_start_pos
            steps = 0
            rewards_collected = 0
            total_reward = 0 
            
            self._gen_grid(self.grid.width, self.grid.height)
            current_quadrant = self.quadrants.index(self.current_quadrant)

            
            while steps < max_steps and rewards_collected < 1:
                state = self.encode_state(*current_pos) + current_quadrant

                if state not in self.q_table_rl_agent:
                    self.q_table_rl_agent[state] = {0: 0, 1: 0, 2: 0, 3: 0}

                if random.uniform(0, 1) < epsilon:
                    action = random.choice(list(self.q_table_rl_agent[state].keys()))
                else:
                    action = max(self.q_table_rl_agent[state], key=self.q_table_rl_agent[state].get)

                new_pos = self.next_position(current_pos, action)
                new_state = self.encode_state(*new_pos) + current_quadrant

                if new_state not in self.q_table_rl_agent:
                    self.q_table_rl_agent[new_state] = {0: 0, 1: 0, 2: 0, 3: 0}

                reward = -1

                if new_pos in self.reward_positions:
                    reward += 30 
                    print(f"Recompensa recogida en {new_pos}")
                    rewards_collected += 1
                    self.reward_positions.remove(new_pos)
                    self.grid.set(*new_pos, None)

                    if rewards_collected == 1:
                        successful_episodes += 1 
                        total_reward += reward

                        if steps + 1 < min_steps_to_goal:
                            min_steps_to_goal = steps + 1
                            best_episode2 = episode + 1
                            break

                total_reward += reward

                self.q_table_rl_agent[state][action] += alpha * (reward + gamma * max(self.q_table_rl_agent[new_state].values()) - self.q_table_rl_agent[state][action])
                current_pos = new_pos
                self.agent_pos = current_pos
                epsilon = max(min_epsilon, epsilon * decay_rate)

                self.render()
                steps += 1
                
                if steps >= max_steps:
                    break


        print(f"\nEntrenamiento completado con {successful_episodes} episodios exitosos de {episodes}")
        self.save_q_table(self.q_table_rl_agent, "q_table21.txt")
        if rewards_collected == 1:
                done = True
                print(f"Se lograron recoger las 1 recompensas en el episodio {episode}.")
        else:
                print("No se lograron recoger las 1 recompensas en ningún episodio.")
        return rewards_collected

    def run_agents(self):
        """Ejecuta a los agentes Empowerment y RL en el ambiente combinado."""
        print("Iniciando el ambiente combinado...")
        
        self.q_table_emp_agent = self.load_q_table("q_table_empql21.txt")
        self.q_table_rl_agent = self.load_q_table("q_table21.txt")
        for episode in range(self.total_episodes):
            print(f"\n--- Episodio {episode + 1} ---")
            
            # Resetea por cada episodio
            self.reset_environment()

            self.current_agent_pos = self.rl_agent_start_pos
            self.current_agent_dir = self.rl_agent_start_dir
            
            current_agent = 'rl'
            done = False
            steps = 0
            max_steps_per_episode = 150
            successE = False

            while not done and steps < max_steps_per_episode:

                if self.current_agent == "emp":
                    self.agent_color = "blue"
                else:
                    self.agent_color = "red"  # Color normal para RL
                # Correr agentes
                self.render()
                self.current_agent = current_agent
                print(f"Agente actual: {current_agent}")
                print(f"Posición actual: {self.current_agent_pos}")
                if current_agent == 'rl':
                    self.rl_agent_start_pos = self.current_agent_pos
                    success = self.q_rl_agent(episodes=1, max_steps=100)
                    self.current_agent_pos = self.agent_pos
                    if self.remaining_rewards == 0 or self.collected_rewards == 1 or success == 1: 
                        current_agent = 'emp'
                        print("Cambio de agente a Empowerment.")
                else:
                    print(f"color del agente: {self.agent_color}")
                    self.emp_agent_start_pos = self.current_agent_pos
                    successE = self.q_emp_agent(episodes=1, max_steps=100)
                    self.current_agent_pos = self.agent_pos
                    if successE == True:
                        print("Cambio de agente a RL.")
                        current_agent = 'rl'
                        self.rl_agent_start_pos = self.current_agent_pos
                        self.current_agent = 'rl'
                        success = self.q_rl_agent(episodes=1, max_steps=100)
                        self.current_agent_pos = self.agent_pos

                steps += 1

                # Fin de episodio
                if steps >= max_steps_per_episode or (success == 1 and successE == True):
                    done = True
                    print(f"Episodio {episode + 1} completado. Bloques (Reloads): {steps}")
                    break

        # Guardar Q-tables
        self.save_q_table(self.q_table_rl_agent, "q_table21.txt")
        self.save_q_table(self.q_table_emp_agent, "q_table_empql21.txt")

def main():
    env = CombinedEnv(render_mode="human", emp_agent_num_pasos=6, rl_agent_num_pasos=6)
    env.run_agents()
    env.close()

if __name__ == "__main__":
    main()