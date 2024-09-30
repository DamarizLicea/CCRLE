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

def calculate_empowerment(state, transition_counts):
    """
    Calcula el empoderamiento usando tanto la distribución marginal de los futuros estados
    como la información mutua entre las acciones y los futuros estados.
    
    :param state: El estado actual del taxi.
    :param transition_counts: Matriz 3D (state_space x action_space x future_state_space) de las transiciones.
    :return: El empowerment total.
    """
    state_slice = transition_counts[state]  # shape: (action_space, future_state_space)
    
    # Distribución marginal de las acciones: sumamos sobre los estados futuros
    action_marginal = np.sum(state_slice, axis=1)  # shape: (action_space,)
    
    # Distribución marginal de los estados futuros: sumamos sobre las acciones
    future_state_marginal = np.sum(state_slice, axis=0)  # shape: (future_state_space,)
    
    # Calcular empowerment basado en la distribución de futuros estados
    nonzero_future_probs = future_state_marginal[future_state_marginal > 0]
    future_state_empowerment = -np.sum(nonzero_future_probs * np.log2(nonzero_future_probs))
    
    # Probabilidad condicional p(future_state | action)
    prob_state_given_action = state_slice / action_marginal[:, None]
    
    # Empowerment basado en la información mutua entre acciones y futuros estados
    action_empowerment = 0
    for action_prob, action_transitions in zip(action_marginal, prob_state_given_action):
        if action_prob > 0:  # Solo incluir acciones con probabilidad positiva
            nonzero_probs = action_transitions[action_transitions > 0]
            action_empowerment += action_prob * np.sum(nonzero_probs * np.log2(nonzero_probs))
    
    # Empowerment total: combinamos ambos términos
    empowerment = future_state_empowerment + (-action_empowerment)
    
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

    transition_counts = np.zeros((env.observation_space.n, env.action_space.n, env.observation_space.n))

    learning_rate = 0.01
    discount_rate = 0.95
    num_episodes = 3500
    max_steps = 90
    epsilon = 0.9  # Inicialmente alta para mayor exploración
    min_epsilon = 0.01
    decay_rate = 0.995  # Tasa de decaimiento de epsilon

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
            empowerment = calculate_empowerment(state, transition_counts)
            
            # Imprimir estado y empowerment
            print(f"Paso: {step}, Acción: {actions[action]}, Nuevo estado: ({taxi_row}, {taxi_col}), Pasajero: {passenger}, Destino: {destination}, Empowerment: {empowerment}")
            
            # Actualizar los conteos de transiciones
            transition_counts[state, action, new_state] += 1
            
            # Calcular distancia al destino y sumar a la recompensa
            distance = calculate_distance_to_destination(env, new_state, destination_coords)
            reward -= distance  # Penalización por distancia

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
                reward -= 1  # Penalización por cada paso

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
