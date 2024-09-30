import numpy as np
import gymnasium as gym

def train_agent(env, q_table, alpha, gamma, epsilon, episodes=1000):
    successful_deliveries = 0  # Contador de entregas exitosas

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        total_rewards = 0  # Acumulador de recompensas para cada episodio

        while not (done or truncated):
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state, :])

            next_state, reward, done, truncated, info = env.step(action)

            # Actualizar Q-Table
            q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action])

            state = next_state
            total_rewards += reward

            # Verificar si se ha realizado una entrega exitosa
            if reward == 20:  # 20 es la recompensa estándar por una entrega exitosa
                successful_deliveries += 1
                print(f"¡Entrega exitosa en el episodio {episode + 1}!")

        # Decaimiento de epsilon
        epsilon = max(0.01, epsilon * 0.995)

        # Imprimir información de cada episodio
        if (episode + 1) % 100 == 0:  # Cada 100 episodios
            print(f"Episodio {episode + 1}: Recompensa total = {total_rewards}, Entregas exitosas = {successful_deliveries}")

    print(f"Total de entregas exitosas después de {episodes} episodios: {successful_deliveries}")
    return q_table

if __name__ == "__main__":
    env = gym.make('Taxi-v3')
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    alpha = 0.1
    gamma = 0.99
    epsilon = 1.0

    q_table = train_agent(env, q_table, alpha, gamma, epsilon, episodes=7500)