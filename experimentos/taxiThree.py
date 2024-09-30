import numpy as np
import gymnasium as gym
import time
import requests
import telegram
import asyncio
import os
from infoTelegram import TOKEN, CHAT_ID
import matplotlib.pyplot as plt

# Definición de las acciones
actions = {
    0: "Sur",
    1: "Norte",
    2: "Este",
    3: "Oeste",
    4: "Recoger",
    5: "Dejar"
}

# Enviar mensajes a Telegram
async def send_telegram_message(message):
    bot = telegram.Bot(token=TOKEN)
    await bot.send_message(chat_id=CHAT_ID, text=message)

# Visualización de la Q-Table
def visualize_qtable(qtable, episode):
    plt.figure(figsize=(10, 10))
    plt.imshow(qtable, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title(f'Q-Table Heatmap at Episode {episode}')
    plt.xlabel('Actions')
    plt.ylabel('States')
    plt.savefig(f'qtable_episode_{episode}.png')
    plt.close()

# Cálculo del Empowerment
def calculate_empowerment(state, transition_counts, n=1):
    empowerment = 0.0
    for action_seq in range(transition_counts.shape[1]):
        total_transitions_action = np.sum(transition_counts[state, action_seq])
        if total_transitions_action == 0:
            continue
        prob_state_given_action = transition_counts[state, action_seq] / total_transitions_action
        for future_state in range(transition_counts.shape[2]):
            total_transitions_state = np.sum(transition_counts[future_state])
            if total_transitions_state == 0:
                continue
            prob_state_action = transition_counts[state, action_seq, future_state] / total_transitions_state
            if prob_state_given_action[future_state] > 0 and prob_state_action > 0:
                empowerment += prob_state_given_action[future_state] * np.log2(prob_state_given_action[future_state] / prob_state_action)
    
    return empowerment

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

# Calcular distancia al destino
def calculate_distance_to_destination(env, state, destination_coords):
    taxi_row, taxi_col, _, _ = env.unwrapped.decode(state)
    dest_row, dest_col = destination_coords
    return abs(taxi_row - dest_row) + abs(taxi_col - dest_col)

# Cargar Q-table
def load_qtable(filename):
    if os.path.exists(filename):
        try:
            return np.load(filename)  
        except EOFError:
            print("El archivo de la Q-table está vacío o corrupto, inicializando nueva Q-table.")
            return np.zeros((500, 6))
    else:
        return np.zeros((500, 6))

# Guardar Q-table
def save_qtable(qtable, filename):
    np.save(filename, qtable)

# Función principal
def main():
    env = gym.make('Taxi-v3', render_mode='human')
    qtable_filename = "qtable_instance_1.npy"
    qtable = load_qtable(qtable_filename)

    transition_counts = np.zeros((env.observation_space.n, env.action_space.n, env.observation_space.n))

    learning_rate = 0.01
    discount_rate = 0.95
    num_episodes = 50
    max_steps = 70
    epsilon = 0.8
    min_epsilon = 0.01
    decay_rate = 0.995

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
            empowerment = calculate_empowerment(state, transition_counts, n=1)

            print(f"Paso: {step}, Acción: {actions[action]}, Nuevo estado: ({taxi_row}, {taxi_col}), Pasajero: {passenger}, Destino: {destination}, Empowerment: {empowerment}")

            # Actualizar conteo de transiciones
            transition_counts[state, action, new_state] += 1

            # Calcular distancia y modificar recompensa
            distance = calculate_distance_to_destination(env, new_state, destination_coords)
            reward -= distance

            # Acciones de recoger y dejar
            if action == 4:  # Recoger
                if passenger != 4 and (taxi_row, taxi_col) == get_destination_coords(passenger):
                    reward += 10  # Recoger correctamente
                    passenger = 4
                else:
                    reward -= 10  # Recoger incorrectamente

            elif action == 5:  # Dejar
                if done and passenger == 4 and (taxi_row, taxi_col) == destination_coords:
                    successful_deliveries += 1
                    print(f"Entrega exitosa al final del episodio {episode}")
                    reward += 50  # Dejar correctamente
                    save_qtable(qtable, qtable_filename)
                    break
                else:
                    reward -= 10  # Dejar incorrectamente

            else:
                reward -= 1  # Penalización por cada paso

            # Actualizar Q-table
            qtable[state, action] = qtable[state, action] + learning_rate * (reward + discount_rate * np.max(qtable[new_state, :]) - qtable[state, action])

            state = new_state
            total_rewards += reward

            if done or truncated:
                break

        np.save(qtable_filename, qtable)

        # Imprimir estado final del episodio
        print(f"Episode: {episode}, Recompensa total: {total_rewards}, Entregas exitosas hasta ahora: {successful_deliveries}")

        epsilon = max(min_epsilon, epsilon * decay_rate)

    end_time = time.time()
    print(f"Tiempo total de entrenamiento: {end_time - start_time} segundos")
    print(f"Entregas exitosas: {successful_deliveries}")

    message = f"Entrenamiento completado. Tiempo: {end_time - start_time} segundos. Entregas exitosas: {successful_deliveries}"
    asyncio.run(send_telegram_message(message))

if __name__ == "__main__":
    main()
