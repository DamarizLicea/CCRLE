import numpy as np
import gymnasium as gym
import time
import telegram
import asyncio
import os
from infoTelegram import TOKEN, CHAT_ID
import pickle
import matplotlib.pyplot as plt

# Definición de las acciones y empowerment
actions = {0: "Sur", 1: "Norte", 2: "Este", 3: "Oeste", 4: "Recoger", 5: "Dejar"}

# Función para enviar mensajes a Telegram
async def send_telegram_message(message):
    bot = telegram.Bot(token=TOKEN)
    await bot.send_message(chat_id=CHAT_ID, text=message)

    
def filter_invalid_actions_by_position(y, x):
    """
    Filtra las acciones inválidas basadas en la posición del taxi en el mapa.
    """
    valid_actions = [0, 1, 2, 3, 4, 5]  # Acciones: Sur, Norte, Este, Oeste, Recoger, Dejar

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


def simulate_n_step_transitions(env, state, n):
    """
    Simula las transiciones a n pasos y devuelve los estados alcanzables.
    """
    reached_states = set()

    def simulate_step(current_state, current_n):
        if current_n == 0:
            return
        for action in range(env.action_space.n):
            transitions = env.P[current_state][action]
            for prob, future_state, _, _ in transitions:
                if prob > 0:
                    reached_states.add(future_state)
                    simulate_step(future_state, current_n - 1)

    simulate_step(state, n)
    return reached_states

def calculate_empowerment_n_steps(env, state, n):
    reached_states = simulate_n_step_transitions(env, state, n)
    num_states_reached = len(reached_states)
    
    if num_states_reached > 0:
        empowerment = np.log2(num_states_reached)
    else:
        empowerment = 0

    return empowerment

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

def save_qtable(qtable, filename):
    with open(filename, 'wb') as f:
        pickle.dump(qtable, f)

def main():
    env = gym.make('Taxi-v3', render_mode='human')
    qtable_filename = "qtable_instance_1.npy"
    qtable = load_qtable(qtable_filename)

    max_empowerment = -np.inf  
    max_empowerment_n = 0  # Guardar en qué valor de n se alcanzó el empowerment máximo
    empowerment_history = []  # Lista para almacenar el empowerment en cada paso
    
    n_step_max = 12

    # Ciclo de 1 episodio
    for episode in range(1): 
        state, _ = env.reset()
        done = False

        for n in range(1, n_step_max + 1):  # buscar el empowerment máximo en n pasos
            empowerment = calculate_empowerment_n_steps(env, state, n)
            empowerment_history.append(empowerment)  # actualizar el historial de empowerment

            # es el empowerment máximo?
            if empowerment > max_empowerment:
                max_empowerment = empowerment
                max_empowerment_n = n

            print(f"n: {n}, Empowerment: {empowerment:.4f}")

    print(f"Máximo empowerment: {max_empowerment:.4f} registrado en n = {max_empowerment_n} pasos.")

    plt.plot(range(1, n_step_max + 1), empowerment_history, marker='o')
    plt.title('Empowerment a diferentes números de pasos')
    plt.xlabel('n pasos')
    plt.ylabel('Empowerment')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()