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


    # def calculate_empowerment(env, state, n=2, epsilon=1e-10):
#     """
#     Calcula el empoderamiento utilizando las probabilidades de transición del entorno Taxi.
    
#     :param env: El entorno de Taxi de Gymnasium.
#     :param state: El estado actual del agente.
#     :param n: Número de pasos a considerar (n-step empowerment).
#     :param epsilon: Umbral para evitar logaritmos de 0 o divisiones por 0.
#     :return: El empoderamiento.
#     """
#     empowerment = 0.0
#     action_space = env.action_space.n  # Número de acciones posibles

#     for action in range(action_space):
#         # Obtener las transiciones desde el estado actual para la acción específica
#         transitions = env.unwrapped.P[state][action]
        
#         for prob, next_state, reward, done in transitions:
#             if prob == 0:
#                 continue  # Si la probabilidad es 0, la ignoramos

#             # Obtener las transiciones desde el siguiente estado
#             future_transitions = env.unwrapped.P[next_state]

#             # Calcular la probabilidad marginal del siguiente estado
#             total_future_prob = sum([t[0] for future_action in future_transitions.values() for t in future_action])

#             if total_future_prob == 0:
#                 continue

#             for future_action in future_transitions:
#                 for f_prob, future_state, f_reward, f_done in future_transitions[future_action]:
#                     if f_prob == 0:
#                         continue
                    
#                     # Calcular la probabilidad condicional del siguiente estado dado la acción
#                     prob_conditional = f_prob / total_future_prob
                    
#                     # Evitar valores muy pequeños y logaritmos de 0
#                     if prob_conditional > epsilon and prob > epsilon:
#                         # Aplicar la fórmula de información mutua para calcular la contribución al empowerment
#                         empowerment += prob * f_prob * np.log2(prob_conditional / prob)

#     return empowerment