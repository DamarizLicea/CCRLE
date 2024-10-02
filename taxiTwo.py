import numpy as np
import gymnasium as gym
import time
import requests
import telegram
import asyncio
import os
from infoTelegram import TOKEN, CHAT_ID
import matplotlib.pyplot as plt
import pickle

# Definición de las acciones
actions = {
    0: "Sur",
    1: "Norte",
    2: "Este",
    3: "Oeste",
    4: "Recoger",
    5: "Dejar"
}

# Función para enviar mensajes a Telegram
async def send_telegram_message(message):
    bot = telegram.Bot(token=TOKEN)
    await bot.send_message(chat_id=CHAT_ID, text=message)


def softmax_action_selection(q_values, temperature=1.0):
    exp_values = np.exp(q_values / temperature)
    probs = exp_values / np.sum(exp_values)
    return np.random.choice(len(q_values), p=probs)

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


def calculate_empowerment(state, action, transition_counts):
    total_transitions = np.sum(transition_counts[state, action])
    if total_transitions == 0:
        return 0.0
    entropy = 0.0
    for next_state in range(transition_counts.shape[2]):
        count = transition_counts[state, action, next_state]
        if count > 0:
            probability = count / total_transitions
            entropy -= probability * np.log2(probability)
    return entropy

# Obtener las coordenadas del destino
def get_destination_coords(destination):
    if destination == 0:  # Rojo
        return 0, 0
    elif destination == 1:  # Verde
        return 0, 4
    elif destination == 2:  # Amarillo
        return 4, 0
    elif destination == 3:  # Azul
        return 4, 3
    else:
        print(f"Error: destino desconocido {destination}.")
        return None

# Calcular la distancia Manhattan al destino
def calculate_distance_to_destination(env, state, destination_coords):
    taxi_row, taxi_col, _, _ = env.unwrapped.decode(state)  # Decodificar el estado
    dest_row, dest_col = destination_coords
    return abs(taxi_row - dest_row) + abs(taxi_col - dest_col)

# Cargar Q-Table
def load_qtable(filename):
    if os.path.exists(filename):
        try:
            return np.load(filename)  # Cargar Q-table existente
        except EOFError:
            print("El archivo de la Q-table está vacío o corrupto, inicializando nueva Q-table.")
            return np.zeros((500, 6))  # Nueva Q-table si el archivo está dañado
    else:
        return np.zeros((500, 6))  # Nueva Q-table si no existe el archivo

# Guardar Q-Table
def save_qtable(qtable, filename):
    with open(filename, 'wb') as f:
        pickle.dump(qtable, f)

# Función principal
def main():
    env = gym.make('Taxi-v3', render_mode='human')
    qtable_filename = "qtable_instance_1.npy"
    qtable = load_qtable(qtable_filename)
    state_counts= np.zeros(env.observation_space.n)
    transition_counts = np.zeros((env.observation_space.n, env.action_space.n, env.observation_space.n))

    learning_rate = 0.05
    discount_rate = 0.95
    num_episodes = 3
    max_steps = 90
    epsilon = 0.9  # Inicialmente alta para mayor exploración
    min_epsilon = 0.01
    decay_rate = 0.99  # Tasa de decaimiento de epsilon

    start_time = time.time()  
    successful_deliveries = 0 

    for episode in range(num_episodes):
        state, _ = env.reset() 
        done = False
        total_rewards = 0

        taxi_row, taxi_col, passenger, destination = env.unwrapped.decode(state)
    
        print(f"Episodio: {episode}, Estado inicial: ({taxi_row}, {taxi_col}), Pasajero: {passenger}, Destino: {destination},")

        destination_coords = get_destination_coords(destination)

        for step in range(max_steps):
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(qtable[state, :])

            new_state, reward, done, truncated, info = env.step(action)

            taxi_row, taxi_col, passenger, destination = env.unwrapped.decode(new_state)
            empowerment = calculate_empowerment(state,  action, transition_counts)

            
            # Imprimir estado y empowerment
            print(f"Paso: {step}, Acción: {actions[action]}, Nuevo estado: ({taxi_row}, {taxi_col}), Pasajero: {passenger}, Destino: {destination}, Empowerment: {empowerment}")
            
            # Actualizar los conteos de transiciones
            transition_counts[state, action, new_state] += 1
            
            # Calcular distancia al destino y sumar a la recompensa
            distance = calculate_distance_to_destination(env, new_state, destination_coords)

            # Acciones de recoger y dejar al pasajero
            if action == 4:  # Recoger
                if passenger != 4 and (taxi_row, taxi_col) == get_destination_coords(passenger):
                    reward += 10  # Recompensa por recoger correctamente
                    passenger = 4  # Indica que el pasajero está en el taxi
                else:
                    reward -= 10  # Penalización por recoger incorrectamente

            elif action == 5:  # Dejar
                if done:
                    reward += 50  # Recompensa por dejar correctamente
                    successful_deliveries += 1
                    print(f"Entrega exitosa en el episodio {episode} después de {step} pasos.")
                    save_qtable(qtable, qtable_filename)  # Guardar Q-table
                    break  # Detener el episodio al lograr la entrega
                else:
                    reward -= 10  # Penalización por dejar incorrectamente
            else:
                reward -= 0.5  # Penalización por cada paso

            # Actualizar Q-table
            qtable[state, action] = qtable[state, action] + learning_rate * (reward + discount_rate * np.max(qtable[new_state, :]) - qtable[state, action])

            state = new_state
            total_rewards += reward

            if done or truncated:
                break

        # Guardar la Q-table
        np.save(qtable_filename, qtable)  

        # Decaimiento de epsilon
        epsilon = max(min_epsilon, epsilon * decay_rate)

        print(f"Episode: {episode}, Recompensa total: {total_rewards}")
        print(f"Entregas exitosas hasta ahora: {successful_deliveries}")

    end_time = time.time()
    print(f"Tiempo total de entrenamiento: {end_time - start_time} segundos")
    print(f"Entregas exitosas al hotel: {successful_deliveries}")

    np.save(qtable_filename, qtable)  # Guardar la Q-table al finalizar

    # Enviar mensaje de Telegram al finalizar
    message = f"Entrenamiento completado. Tiempo total de entrenamiento: {end_time - start_time} segundos. Entregas exitosas al hotel: {successful_deliveries}"
    asyncio.run(send_telegram_message(message))

if __name__ == "__main__":
    main()
