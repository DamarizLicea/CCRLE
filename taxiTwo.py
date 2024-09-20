import numpy as np
import gymnasium as gym
import time
import requests
import telegram
import asyncio
import os
from infoTelegram import TOKEN, CHAT_ID

# Definición de las acciones
actions = {
    0: "Sur",
    1: "Norte",
    2: "Este",
    3: "Oeste",
    4: "Recoger",
    5: "Dejar"
}


async def send_telegram_message(message):
    bot = telegram.Bot(token=TOKEN)
    await bot.send_message(chat_id=CHAT_ID, text=message)


def calculate_empowerment(state, transition_counts, n=1):
    """
    Calcula el empoderamiento como la capacidad del canal de la secuencia de acciones y los estados futuros.
    
    :param state: El estado actual del agente.
    :param transition_counts: Un arreglo de NumPy que contiene los conteos de transiciones.
    :param n: Número de pasos a considerar (n-step empowerment).
    :return: El empoderamiento.
    """
    empowerment = 0.0
    for action_seq in range(transition_counts.shape[1]):
        # p(a) = Σ p(s', a) -> Total de transiciones para una acción
        # Cuántas veces una acción particular lleva a cada estado futuro
        total_transitions_action = np.sum(transition_counts[state, action_seq])
        
        if total_transitions_action == 0:
            continue  # Si no hay transiciones, pasa a la siguiente acción

        # Probabilidad de llegar a cualquier estado futuro dado que tienes una secuencia de acciones p(s' | a)
        # p(s' | a) = p(s', a) / p(a) -> Cuantas veces desde un estado particular, se tomaron ciertas acciones / total de veces que hemos tomado esa acción
        prob_state_given_action = transition_counts[state, action_seq] / total_transitions_action
        
        for future_state in range(transition_counts.shape[2]):
            # Σ p(s') -> Suma de todas las transiciones hacia un estado futuro
            total_transitions_state = np.sum(transition_counts[state])
            
            if total_transitions_state == 0:
                continue  # Si no hay transiciones, pasamos al siguiente estado

            # Probabilidad de que el agente termine en un estado futuro específico dado un estado actual y una secuencia de acciones
            # p(s'|s,a) = cuántas veces, desde un estado inicial, se realizó una secuencia de acciones 
            #             y terminó en un estado futuro / total de transiciones que comienzan desde ese estado
            prob_state_action = transition_counts[state, action_seq, future_state] / total_transitions_state
            
            # Si la probabilidad condicional es mayor a cero, entonces calculamos la contribución al empoderamiento
            if prob_state_given_action[future_state] > 0:
                # Fórmula de información mutua: I(S'; A) = Σ p(s' | a) * log2( p(s' | a) / p(s'|s,a) )
                empowerment += prob_state_given_action[future_state] * np.log2(prob_state_given_action[future_state] / prob_state_action)
    
    return empowerment

def get_destination_coords(destination):
    """
    Obtiene las coordenadas de la fila y columna del destino.
    """
    if destination == 0:  # R
        return 0, 0
    elif destination == 1:  # G
        return 0, 4
    elif destination == 2:  # Y
        return 4, 0
    elif destination == 3:  # B
        return 4, 3
    else:
        print(f"Error: destino desconocido {destination}.")
        return None

def calculate_distance_to_destination(env, state, destination_coords):
    """
    Calcula la distancia Manhattan entre el taxi y el destino.
    """
    taxi_row, taxi_col, _, _ = env.unwrapped.decode(state)  # Decodificar el estado
    dest_row, dest_col = destination_coords
    return abs(taxi_row - dest_row) + abs(taxi_col - dest_col)

def load_qtable(filename):
    # Si el archivo no existe, inicializamos una nueva Q-table
    if os.path.exists(filename):
        try:
            return np.load(filename)  # Cargar Q-table existente
        except EOFError:
            print("El archivo de la Q-table está vacío o corrupto, inicializando nueva Q-table.")
            return np.zeros((500, 6))  # Inicializar nueva Q-table si el archivo está dañado
    else:
        return np.zeros((500, 6))  # Inicializar una nueva Q-table si no existe el archivo

def save_qtable(qtable, filename):
    np.save(filename, qtable)


def main():
    env = gym.make('Taxi-v3', render_mode='human')
    qtable_filename = "qtable_instance_1.npy"  # Asegúrate de usar un nombre diferente para cada archivo
    qtable = load_qtable(qtable_filename)

    transition_counts = np.zeros((env.observation_space.n, env.action_space.n, env.observation_space.n))

    learning_rate = 0.01
    discount_rate = 0.99
    num_episodes = 4
    max_steps = 100
    epsilon = 1.0  # Inicialmente alta para mayor exploración
    min_epsilon = 0.01
    decay_rate = 0.995  # Tasa de decaimiento de epsilon

    start_time = time.time()  # Iniciar el temporizador
    successful_deliveries = 0  # Contador de entregas exitosas

    for episode in range(num_episodes):
        state, _ = env.reset() 
        done = False
        total_rewards = 0

        taxi_row, taxi_col, passenger, destination = env.unwrapped.decode(state)
        print(f"Episodio: {episode}, Estado inicial: ({taxi_row}, {taxi_col}), Pasajero: {passenger}, Destino: {destination}")

        destination_coords = get_destination_coords(destination)

        # Verificar que destination_coords no sea None
        if destination_coords is None:
            print(f"Error en el episodio {episode}: destino inválido {destination}. Reiniciando el episodio.")
            continue  # Saltar este episodio si no hay destino válido

        for step in range(max_steps):
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(qtable[state, :])

            new_state, reward, done, truncated, info = env.step(action)

            taxi_row, taxi_col, passenger, destination = env.unwrapped.decode(new_state)
            print(f"Paso: {step}, Acción: {actions[action]}, Nuevo estado: ({taxi_row}, {taxi_col}), Pasajero: {passenger}, Destino: {destination}")
            #Recuerda que passenger en 4 es dentro del taxi
            # Actualizar los conteos de transiciones


            transition_counts[state, action, new_state] += 1
            distance = calculate_distance_to_destination(env, new_state, destination_coords)
            empowerment = calculate_empowerment(state, transition_counts, n=1)

            reward += empowerment - distance 

            # Reward Engineering ajustado
            if action == 4:  # Recoger
                if (taxi_row, taxi_col) == get_destination_coords(passenger):
                    reward += 10  # Recompensa por recoger correctamente
                    passenger = 4
                else:
                    reward -= 10  # Penalización por intentar recoger incorrectamente
            elif action == 5:  # Dejar
                if passenger == 4 and (taxi_row, taxi_col) == destination_coords:
                    reward += 50  # Recompensa por dejar correctamente
                    successful_deliveries += 1  # Incrementar el contador de entregas exitosas

                    message = f"Entrega exitosa al hotel en el episodio {episode} después de {step} pasos."
                    asyncio.run(send_telegram_message(message))
                    print(message)

                    save_qtable(qtable, qtable_filename)

                    return  # Detener el programa
                else:
                    reward -= 10  # Penalización por intentar dejar incorrectamente
            else:
                reward -= 1  # Penalización por cada paso para incentivar la eficiencia

            # Aumentar la recompensa en función del empoderamiento
            reward += empowerment  # Recompensa basada en la capacidad de acción futura

            # Actualizar Q-table
            qtable[state, action] = qtable[state, action] + learning_rate * (reward + discount_rate * np.max(qtable[new_state, :]) - qtable[state, action])

            state = new_state
            total_rewards += reward  # Acumular la recompensa total

            if done or truncated:
                break

        np.save(qtable_filename, qtable)  # Guardar la Q-table después de cada episodio

        # Decaimiento de epsilon
        epsilon = max(min_epsilon, epsilon * decay_rate)

        print(f"Episode: {episode}, Total Reward: {total_rewards}")
        print(f"Paso: {step}, Recompensa: {reward}")

    end_time = time.time()
    print(f"Tiempo total de entrenamiento: {end_time - start_time} segundos")
    print(f"Entregas exitosas al hotel: {successful_deliveries}")

    np.save(qtable_filename, qtable)  # Guardar la Q-table al finalizar el entrenamiento

    # Enviar mensaje de Telegram al finalizar
    message = f"Entrenamiento completado. Tiempo total de entrenamiento: {end_time - start_time} segundos. Entregas exitosas al hotel: {successful_deliveries}"
    asyncio.run(send_telegram_message(message))

if __name__ == "__main__":
    main()