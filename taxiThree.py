import numpy as np
import pandas as pd
import gymnasium as gym
import time

# Diccionario para mapear acciones a palabras
action_dict = {
    0: "Sur",
    1: "Norte",
    2: "Este",
    3: "Oeste",
    4: "Recoger",
    5: "Dejar"
}

def main():
    env = gym.make('Taxi-v3', render_mode='human')
    qtable = np.zeros((env.observation_space.n, env.action_space.n))
    learning_rate = 0.1
    discount_rate = 0.99
    num_episodes = 10
    max_steps = 200
    epsilon = 1.0  # Inicialmente alta para mayor exploración
    min_epsilon = 0.01
    decay_rate = 0.995  # Tasa de decaimiento de epsilon

    start_time = time.time()  # Iniciar el temporizador

    for episode in range(num_episodes):
        state, _ = env.reset()  # Asegúrate de que state es un entero
        done = False
        total_rewards = 0
        print(f"Episodio {episode + 1}")

        for step in range(max_steps):
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(qtable[state, :])

            new_state, reward, terminated, truncated, info = env.step(action)

            qtable[state, action] = qtable[state, action] + learning_rate * (reward + discount_rate * np.max(qtable[new_state, :]) - qtable[state, action])

            state = new_state
            total_rewards += reward

            # Imprimir la acción en palabras
            print(f"Paso {step + 1}: Acción {action_dict[action]}, Recompensa {reward}, Nuevo Estado {new_state}")

            if terminated or truncated:
                done = True
                break

        # Reducir epsilon
        epsilon = max(min_epsilon, epsilon * decay_rate)

        print(f"Recompensas totales en el episodio {episode + 1}: {total_rewards}\n")

    # Guardar la tabla Q resultante en un archivo
    np.save("qtable.npy", qtable)
    print("Tabla Q guardada en 'qtable.npy'")

    env.close()

    end_time = time.time()  # Detener el temporizador
    training_time = end_time - start_time
    print(f"El entrenamiento tomó {training_time:.2f} segundos")

    # Crear un DataFrame de pandas con la tabla Q
    df_qtable = pd.DataFrame(qtable)

    # Renombrar las columnas con las acciones
    df_qtable.columns = [action_dict[i] for i in range(df_qtable.shape[1])]

    # Agregar una columna para los estados
    df_qtable.index.name = 'Estado'

    # Mostrar el DataFrame
    print(df_qtable)

    # Guardar el DataFrame en un archivo CSV para una mejor visualización
    df_qtable.to_csv("qtable_formatted.csv")
    print("Tabla Q formateada guardada en 'qtable_formatted.csv'")

if __name__ == "__main__":
    main()