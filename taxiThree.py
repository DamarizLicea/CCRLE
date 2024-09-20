import numpy as np
import gymnasium as gym
import time
import requests
import telegram
import asyncio
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


def main():
    env = gym.make('Taxi-v3', render_mode='human')
    qtable = np.zeros((env.observation_space.n, env.action_space.n))
    transition_counts = np.zeros((env.observation_space.n, env.action_space.n, env.observation_space.n))
    state_counts = np.zeros(env.observation_space.n)

    learning_rate = 0.01
    discount_rate = 0.99
    num_episodes = 7000
    max_steps = 100
    epsilon = 1.0  # Inicialmente alta para mayor exploración
    min_epsilon = 0.01
    decay_rate = 0.995  # Tasa de decaimiento de epsilon

    start_time = time.time()  # Iniciar el temporizador
    successful_deliveries = 0  # Contador de entregas exitosas

    for episode in range(num_episodes):
        state, _ = env.reset()  # Asegúrate de que state es un entero
        done = False
        total_rewards = 0

        for step in range(max_steps):
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(qtable[state, :])

            new_state, reward, done, truncated, info = env.step(action)

            # Actualizar los conteos de transiciones
            transition_counts[state, action, new_state] += 1
            
            # Reward Engineering
            if action == 4:  # Recoger
                if info.get('passenger') == 'taxi':
                    reward += 10  # Recompensa por recoger correctamente
                else:
                    reward -= 5  # Penalización por intentar recoger incorrectamente
            elif action == 5:  # Dejar
                if info.get('passenger') == 'destination':
                    reward += 20  # Recompensa por dejar correctamente
                    successful_deliveries += 1  # Incrementar el contador de entregas exitosas
                    print(successful_deliveries)

                    # Enviar mensaje de Telegram y detener el programa
                    message = f"Entrega exitosa al hotel en el episodio {episode} después de {step} pasos."
                    asyncio.run(send_telegram_message(message))
                    print(message)
                    return  # Detener el programa
                else:
                    reward -= 10  # Penalización por intentar dejar incorrectamente
            else:
                reward -= 1  # Penalización por cada paso para incentivar la eficiencia

            # Actualizar Q-table
            qtable[state, action] = qtable[state, action] + learning_rate * (reward + discount_rate * np.max(qtable[new_state, :]) - qtable[state, action])

            state = new_state
            total_rewards += reward  # Acumular la recompensa total

            if done or truncated:
                break

        # Decaimiento de epsilon
        epsilon = max(min_epsilon, epsilon * decay_rate)

        print(f"Episode: {episode}, Total Reward: {total_rewards}")
        print(f"Paso: {step}, Recompensa: {reward}")

    end_time = time.time()
    print(f"Tiempo total de entrenamiento: {end_time - start_time} segundos")
    print(f"Entregas exitosas al hotel: {successful_deliveries}")

    # Enviar mensaje de Telegram al finalizar
    message = f"Entrenamiento completado. Tiempo total de entrenamiento: {end_time - start_time} segundos. Entregas exitosas al hotel: {successful_deliveries}"
    asyncio.run(send_telegram_message(message))

if __name__ == "__main__":
    main()