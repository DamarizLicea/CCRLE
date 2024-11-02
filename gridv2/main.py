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
    """ Clase para crear muros que se pueden atravesar (pasillo) """
    def can_overlap(self):
        return True

class SimpleEnv(MiniGridEnv):
    def __init__(self, size=13, agent_start_pos=(1, 1), agent_start_dir=0, max_steps: int | None = None, **kwargs):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        mission_space = MissionSpace(mission_func=self._gen_mission)
        self.current_quadrant = 0
        self.reward_positions = [(3, 3), (3, 9), (9, 3), (9, 9)] 

        if max_steps is None:
            max_steps = 4 * size**2
        
        # Inicializar el entorno
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
        return "Obtener las recompensas"

    def _gen_grid(self, width, height):
        """Generar el tablero, poner pasillos y recompensas en los cuadrantes."""
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Pasillos 
        for i in range(1, height - 1):
            self.grid.set(6, i, PassableWall())
        for i in range(1, width - 1):
            self.grid.set(i, 6, PassableWall())

        self.place_reward()

        # Colocar al agente
        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir
        self.mission = "Obtener las recompensas"

    def place_reward(self):
        """
        Poner la recompensa en el cuadrante.
        """
        x, y = self.reward_positions[self.current_quadrant]
        self.put_obj(Goal(), x, y)

    def update_reward_location(self):
        """Actualiza la ubicación de la recompensa al siguiente cuadrante."""
        self.current_quadrant = (self.current_quadrant + 1) % 4
        self._gen_grid(self.grid.width, self.grid.height)

    def encode_state(self, x, y):
        """Codificar los estados a numeros enteros."""
        return (x * 100 + y) + self.current_quadrant  
            # Posicion x del agente * 100 + Posicion y del agente + Cuadrante actual de la recompensa

    def calculate_empowerment(self, x, y, n_steps=5):
        """ Calculo del empowerment a n pasos a futuro desde la posición actual 
         params:
            x (int): Posición x
            y (int): Posición y
            n_steps (int): Número de pasos a futuro   
        """
        # Acuerdte que set evita duplicados
        reachable_states = set()
        
        def explore_positions(pos, steps):
            """ Explorar las posiciones alcanzables a n pasos """
            if steps == 0:
                return
            for direction in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # Arriba, derecha, abajo, izquierda
                new_x, new_y = pos[0] + direction[0], pos[1] + direction[1]
                # Existe la celda y no es una pared
                if 0 <= new_x < self.grid.width and 0 <= new_y < self.grid.height:
                    cell = self.grid.get(new_x, new_y)
                    if not isinstance(cell, Wall) or isinstance(cell, PassableWall):
                        # Si la celda no ha sido visitada, agregarla a las celdas alcanzables
                        if (new_x, new_y) not in reachable_states:
                            reachable_states.add((new_x, new_y))
                            explore_positions((new_x, new_y), steps - 1)

        explore_positions((x, y), n_steps)
        
        return np.log2(len(reachable_states) + 1)

    def calculate_empowerment_matrix(self):
        """ Matriz de empowerment para cada celda en el entorno """
        for row in range(1, self.grid.height - 1):
            for col in range(1, self.grid.width - 1):
                if not isinstance(self.grid.get(col, row), Wall): # No calcular para las paredes ni pasillos
                    self.empowerment_matrix[row, col] = self.calculate_empowerment(col, row)

    def save_empowerment_matrix(self, filename="empowerment_matrix.txt"):
        """ Guardar la matriz de empowerment en un archivo de texto """
        np.savetxt(filename, self.empowerment_matrix, fmt="%.2f")

    def auto_move(self):
        """ Función para mover al agente 1 sin usar el teclado """
        for row in range(1, self.grid.height - 1):
            for col in range(1, self.grid.width - 1):
                self.agent_pos = (col, row)
                self.agent_dir = 0
                self.render()
                # Imprimir info del agente
                print(f"Agente en posición (x, y): ({col}, {row}) | Estado codificado: {self.encode_state(col, row)}")
                # Pequeña pausa para poder ver el movimiento
                time.sleep(0.1)

        center_x, center_y = self.grid.width // 2, self.grid.height // 2
        self.agent_pos = (center_x, center_y)
        self.agent_dir = 0
        self.render()


    # ------------------ Q-learning ------------------ #
    def initialize_q_table(self):
        self.q_table = {}
        for row in range(1, self.grid.height - 1):
            for col in range(1, self.grid.width - 1):
                state = self.encode_state(col, row)
                self.q_table[state] = {0: 0, 1: 0, 2: 0, 3: 0}

    def save_q_table(self, filename="q_table.txt"):
        with open(filename, "w") as file:
            for state, actions in self.q_table.items():
                file.write(f"State {state}: {actions}\n")

    
    #Tantos pasos porque no estoy cortando en episodios.
    
    def q_learning_agent(self, alpha=0.1, gamma=0.9, epsilon=0.1, max_steps=1700):
        """Agente que usa RL (Q-learning) para encontrar las recompensas."""
        current_pos = self.agent_start_pos
        steps = 0
        found_goal = False
        
        while steps < max_steps:
            state = self.encode_state(*current_pos)
            if state not in self.q_table:
                self.q_table[state] = {0: 0, 1: 0, 2: 0, 3: 0}

            # Exploración o explotación
            if random.uniform(0, 1) < epsilon:
                action = random.choice(list(self.q_table[state].keys()))
            else:
                action = max(self.q_table[state], key=self.q_table[state].get)

            new_pos = self.next_position(current_pos, action)
            new_state = self.encode_state(*new_pos)

            if new_state not in self.q_table:
                self.q_table[new_state] = {0: 0, 1: 0, 2: 0, 3: 0}

            # Alcanzó la recompensa?
            if new_pos == self.reward_positions[self.current_quadrant]:
                reward = 1
                found_goal = True
                current_quadrant = self.current_quadrant
                print(f"Recompensa encontrada en el cuadrante {current_quadrant}")
                self.update_reward_location()  # Mover la recompensa al siguiente cuadrante
            else:
                reward = -0.1
                found_goal = False

            # Actualización de Q-table
            self.q_table[state][action] += alpha * (reward + gamma * max(self.q_table[new_state].values()) - self.q_table[state][action])
            current_pos = new_pos
            self.agent_pos = current_pos
            self.render()
            steps += 1

            if found_goal:
                found_goal = False
                continue
        # Truncado
        print("Entrenamiento completado o límite de pasos alcanzado.")
        self.save_q_table()


    def next_position(self, pos, action):
        """Calcular la siguiente posición basada en la acción."""
        direction = [(0, 1), (1, 0), (0, -1), (-1, 0)][action]
        new_x, new_y = pos[0] + direction[0], pos[1] + direction[1]
        
        if 0 <= new_x < self.grid.width and 0 <= new_y < self.grid.height:
            if not isinstance(self.grid.get(new_x, new_y), Wall) or isinstance(self.grid.get(new_x, new_y), PassableWall):
                return new_x, new_y
        
        return pos

    def run_agents(self):
        """Programa para ordenar la ejecución de los agentes."""
        self.reset()
        print("Ejecutando el primer agente (empowerment)...")
        self.auto_move()
        print("Primer agente terminó su recorrido.\nIniciando el agente de Q-learning...")
        
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
