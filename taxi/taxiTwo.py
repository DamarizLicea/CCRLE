import numpy as np
import gymnasium as gym
import time
import telegram
import asyncio
import os
from infoTelegram import TOKEN, CHAT_ID
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

def simulate_n_step_transitions(env, state, n):
    """
    Simula las transiciones a n pasos y devuelve los estados alcanzables, sin contar estados repetidos.
    params:
        env: entorno de Gym
        state: estado inicial
        n: número de pasos
    """
    # Almacena los estados alcanzables (set para evitar repetidos)
    reached_states = set()

    def simulate_step(current_state, current_n, visited_states):
        if current_n == 0:
            return
        # action space: 0, 1, 2, 3, 4, 5
        for action in range(env.action_space.n):
            transitions = env.P[current_state][action]
            # env.P[state][action] = [(probabilidad, estado futuro, recompensa, done), ...]
            for prob, future_state, _, _ in transitions:
                if prob > 0 and future_state not in visited_states:
                    # Añadir el estado futuro a los estados alcanzables solo si no se ha visitado antes
                    reached_states.add(future_state)
                    visited_states.add(future_state)
                    # Llamada recursiva para simular el siguiente paso
                    simulate_step(future_state, current_n - 1, visited_states)

    simulate_step(state, n, visited_states={state})  # Inicializamos con el estado actual ya visitado
    return reached_states

def calculate_empowerment_n_steps(env, state, n):
    """
    Calcula el empowerment a n pasos como el logaritmo del número de estados únicos alcanzables.
    params:
        env: entorno de Gym
        state: estado inicial
        n: número de pasos
    """
    # Estados alcanzables a n pasos (sin repetidos)
    reached_states = simulate_n_step_transitions(env, state, n)
    num_states_reached = len(reached_states)
    
    if num_states_reached > 0:
        empowerment = np.log2(num_states_reached)
    else:
        empowerment = 0

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

# Cargar Q-Table
def load_qtable(filename):
    if os.path.exists(filename):
        try:
            return np.load(filename, allow_pickle=True)
        except EOFError:
            print("El archivo de la Q-table está vacío o corrupto, inicializando nueva Q-table.")
            return np.zeros((500, 6)) 
    else:
        return np.zeros((500, 6))

# Guardar Q-Table
def save_qtable(qtable, filename):
    with open(filename, 'wb') as f:
        pickle.dump(qtable, f)

# Función principal
def main():
    env = gym.make('Taxi-v3', render_mode='human')
    qtable_filename = "qtable_instance_1.npy"
    qtable = load_qtable(qtable_filename)
    state_counts = np.zeros(env.observation_space.n)
    transition_counts = np.zeros((env.observation_space.n, env.action_space.n, env.observation_space.n))

    learning_rate = 0.05
    discount_rate = 0.95
    num_episodes = 3
    max_steps = 80
    epsilon = 0.9
    min_epsilon = 0.01
    decay_rate = 0.99 

    start_time = time.time()  
    successful_deliveries = 0 

    for episode in range(num_episodes):
        state, _ = env.reset() 
        done = False
        total_rewards = 0

        taxi_row, taxi_col, passenger, destination = env.unwrapped.decode(state)
    
        print(f"Episodio: {episode}, Estado inicial: ({taxi_row}, {taxi_col}), Pasajero: {passenger}, Destino: {destination},")
        
        for step in range(max_steps):
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(qtable[state, :])

            new_state, reward, done, truncated, info = env.step(action)

            taxi_row, taxi_col, passenger, destination = env.unwrapped.decode(new_state)
            empowerment = calculate_empowerment_n_steps(env, new_state, n=5)
        
            print(f"Paso: {step}, Acción: {actions[action]}, Nuevo estado: ({taxi_row}, {taxi_col}), Pasajero: {passenger}, Destino: {destination}, Empowerment: {empowerment}")
            
            # Actualizar los conteos de transiciones
            transition_counts[state, action, new_state] += 1

            # Reward engineering
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
        np.save(qtable_filename, qtable)  

        # Decaimiento de epsilon
        epsilon = max(min_epsilon, epsilon * decay_rate)

        print(f"Episode: {episode}, Recompensa total: {total_rewards}")
        print(f"Entregas exitosas hasta ahora: {successful_deliveries}")

    end_time = time.time()
    print(f"Tiempo total de entrenamiento: {end_time - start_time} segundos")
    print(f"Entregas exitosas al hotel: {successful_deliveries}")

    np.save(qtable_filename, qtable)

    # ------------------ TELEGRAM ------------------ #
    message = f"Entrenamiento completado. Tiempo total de entrenamiento: {end_time - start_time} segundos. Entregas exitosas al hotel: {successful_deliveries}"
    asyncio.run(send_telegram_message(message))

if __name__ == "__main__":
    main()
