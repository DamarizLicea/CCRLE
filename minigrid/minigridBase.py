import numpy as np
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall
from minigrid.minigrid_env import MiniGridEnv

class SimpleQEnv(MiniGridEnv):
    def __init__(self, size=10, agent_start_pos=(1, 1), agent_start_dir=0, max_steps=200, **kwargs):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        self.remaining_rewards = 0

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "collect all rewards"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir

        # Colocar recompensas aleatorias
        self.place_random_rewards()

        self.mission = "collect all rewards"

    def place_random_rewards(self):
        """
        Coloca varias recompensas de forma aleatoria en la cuadrícula.
        """
        num_rewards = 3 
        self.remaining_rewards = num_rewards
        for _ in range(num_rewards):
            reward = Goal()
            self.place_obj(reward)

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)


        obj = self.grid.get(*self.agent_pos)
        if obj is not None and isinstance(obj, Goal):
            reward = 1 
            self.remaining_rewards -= 1
            self.grid.set(*self.agent_pos, None)

        if self.remaining_rewards == 0:
            self.place_random_rewards()

        done = False

        return obs, reward, done, truncated, info

def state_to_index(pos, dir, env):
    """
    Convierte la posición y dirección del agente en un índice único para la tabla Q.
    """
    return (pos[0] * env.width + pos[1]) * 4 + dir

def q_learning(env, num_episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.9, epsilon_decay=0.99):
    q_table = np.zeros([env.width * env.height * 4, env.action_space.n])  # 4 direcciones

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            state_index = state_to_index(env.agent_pos, env.agent_dir, env)

            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Exploración
            else:
                action = np.argmax(q_table[state_index])  # Explotación

            next_obs, reward, done, truncated, _ = env.step(action)

            next_state_index = state_to_index(env.agent_pos, env.agent_dir, env)
            best_future_q = np.max(q_table[next_state_index])
            q_table[state_index, action] += alpha * (reward + gamma * best_future_q - q_table[state_index, action])

            total_reward += reward
            step_count += 1

            print(f"Paso: {step_count}  Estado: {env.agent_pos}, Dirección: {env.agent_dir}, Acción: {action}, Recompensa: {reward}, Recompensas restantes: {env.remaining_rewards}")

            if truncated or env.remaining_rewards == 0:
                break

        epsilon *= epsilon_decay
        print(f"Episodio: {episode}, Recompensa total: {total_reward}, Epsilon: {epsilon}")

    return q_table


def main():
    env = SimpleQEnv(render_mode="human")

    q_table = q_learning(env, num_episodes=1)

    obs = env.reset()
    print(obs)
    exit()
    done = False

    # agregar direccion a la qtable
    #max step 1 
    # pq no guarda los numeros
    
    while not done:
        state_index = env.agent_pos[0] * env.width + env.agent_pos[1]
        action = np.argmax(q_table[state_index])
        obs, reward, done, truncated, info = env.step(action)

        action_descriptions = {0: "Mover adelante", 1: "Girar derecha", 2: "Girar izquierda", 3: "Recoger", 4: "Abrir puerta"}
        print(f"Movimiento: Estado: {env.agent_pos}, Acción: {action_descriptions[action]}, Recompensa: {reward}, Recompensas restantes: {env.remaining_rewards}")

        env.render()

        if truncated or env.remaining_rewards == 0:
            break


if __name__ == "__main__":
    main()

    