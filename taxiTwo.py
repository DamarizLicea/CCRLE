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


def filter_invalid_actions_by_position(y,x):
    """
    Filtra las acciones inválidas basadas en la posición del taxi en el mapa.
    :param x: Fila actual del taxi.
    :param y: Columna actual del taxi.
    :return: Lista de acciones válidas.
    """
    # Mapa de acciones: 0: Sur, 1: Norte, 2: Este, 3: Oeste
    valid_actions = [0, 1, 2, 3, 4, 5]  # Acciones: Sur, Norte, Este, Oeste, Recoger, Dejar

    # Limitar movimiento según las coordenadas y las paredes
    if y == 0:  # Borde superior
        valid_actions.remove(1)  # No puede moverse al norte
    if y == 4:  # Borde inferior
        valid_actions.remove(0)  # No puede moverse al sur
    if x == 0:  # Borde izquierdo
        valid_actions.remove(3)  # No puede moverse al oeste
    if x == 4:  # Borde derecho
        valid_actions.remove(2)  # No puede moverse al este

    # Paredes adicionales según el entorno de Taxi
    if (x == 2 and y == 0) or (x == 2 and y == 1) or (x == 1 and y == 3) or (x == 1 and y == 4) or (x == 3 and y == 3) or (x == 3 and y == 4):
        valid_actions.remove(3)  # No puede moverse al oeste en esas posiciones
    if (x == 1 and y == 0) or (x == 1 and y == 1) or (x == 2 and y == 3) or (x == 2 and y == 4) or (x == 0 and y == 3) or (x == 0 and y == 4):
        valid_actions.remove(2)  # No puede moverse al este

    return valid_actions


# def calculate_marginal_distributions(env, state):
#     """
#     Calcula las distribuciones marginales P(future_state|state) y P(action|state).
    
#     :param env: El entorno de Taxi de Gymnasium.
#     :param state: El estado actual.
#     :return: Las distribuciones marginales.
#     """
#     transition_slice = env.P[state]
#     taxi_row, taxi_col, _, _ = env.unwrapped.decode(state)
#     valid_actions = filter_invalid_actions_by_position(taxi_row, taxi_col)
    
#     # crear distribuciones marginales
#     #marginal action es un vector de tamaño 6, uno por cada acción
#     marginal_action = np.zeros(env.action_space.n)  # P(action|state)
#     #marginal future state es un vector de tamaño 500, uno por cada estado
#     marginal_future_state = np.zeros(env.observation_space.n)  # P(future_state|state)
    
#     # Recorre cada acción posible
#     for action, transitions in valid_transitions.items():
#         # total prob action siempre es 1
#         total_prob_action = sum([prob for prob, _, _, _ in transitions])
#         # los valores de probabilidad de cada acción se guardan en el vector marginal_action
#         # por ejemplo si la acción 0 tiene probabilidad 1, entonces en la posición 0 del vector marginal_action se guarda 1
#         marginal_action[action] = total_prob_action
        
#         # Sumar las probabilidades de todos los futuros estados
#         for prob, future_state, _, _ in transitions:
#             marginal_future_state[future_state] += prob

#     return marginal_action, marginal_future_state

def calculate_marginal_distributions(env, state):
    """
    Calcula las distribuciones marginales P(future_state|state) y P(action|state).
    
    :param env: El entorno de Taxi de Gymnasium.
    :param state: El estado actual.
    :return: Las distribuciones marginales.
    """
    transition_slice = env.P[state]  # Transiciones para el estado actual
    taxi_row, taxi_col, _, _ = env.unwrapped.decode(state)  # Decodificar la posición actual del taxi
    valid_actions = filter_invalid_actions_by_position(taxi_row, taxi_col)  # Acciones válidas
    
    # Inicializar distribuciones marginales
    marginal_action = np.zeros(env.action_space.n)  # P(action|state)
    marginal_future_state = np.zeros(env.observation_space.n)  # P(future_state|state)
    
    # Recorre cada acción válida
    for action in valid_actions:
        transitions = transition_slice[action]
        # Total probabilidad de la acción
        total_prob_action = sum([prob for prob, _, _, _ in transitions])
        marginal_action[action] = total_prob_action
        
        # Sumar las probabilidades de los futuros estados
        for prob, future_state, _, _ in transitions:
            marginal_future_state[future_state] += prob

    return marginal_action, marginal_future_state


def calculate_empowerment(env, state, epsilon=1e-10):
    """
    Calcula el empoderamiento utilizando las distribuciones marginales.
    
    :param env: El entorno de Taxi de Gymnasium.
    :param state: El estado actual.
    :param epsilon: Umbral para evitar logaritmos de 0 o divisiones por 0.
    :return: El empoderamiento.
    """
    empowerment = 0.0
    
    # Obtener las distribuciones marginales
    marginal_action, marginal_future_state = calculate_marginal_distributions(env, state)
    
    for action in range(env.action_space.n):
        if marginal_action[action] == 0:  # Acción inválida, saltarla
            continue

        for future_state in range(env.observation_space.n):
            prob_action = marginal_action[action]
            prob_future = marginal_future_state[future_state]

            if prob_action > epsilon and prob_future > epsilon:
                empowerment += prob_action * prob_future * np.log2(prob_future / prob_action)

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
    state_counts= np.zeros(env.observation_space.n)
    transition_counts = np.zeros((env.observation_space.n, env.action_space.n, env.observation_space.n))

    learning_rate = 0.05
    discount_rate = 0.95
    num_episodes = 3
    max_steps = 100
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
            empowerment = calculate_empowerment(env, state, epsilon=1e-10)

            
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
