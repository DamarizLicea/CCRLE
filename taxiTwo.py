import numpy as np
import gymnasium as gym

def main():
    env = gym.make('Taxi-v3', render_mode='human')
    qtable = np.zeros((env.observation_space.n, env.action_space.n))
    learning_rate = 0.1
    discount_rate = 0.99
    num_episodes = 1000
    max_steps = 100
    epsilon = 0.1  # Probabilidad de exploración

    for episode in range(num_episodes):
        state, _ = env.reset()  # Asegúrate de que state es un entero
        done = False
        total_rewards = 0
        print(f"Episode {episode + 1}")

        for step in range(max_steps):
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(qtable[state, :])

            new_state, reward, terminated, truncated, info = env.step(action)

            qtable[state, action] = qtable[state, action] + learning_rate * (reward + discount_rate * np.max(qtable[new_state, :]) - qtable[state, action])

            state = new_state
            total_rewards += reward

            print(f"Step {step + 1}: Action {action}, Reward {reward}, New State {new_state}")

            if terminated or truncated:
                done = True
                break

        print(f"Total rewards in episode {episode + 1}: {total_rewards}\n")

    env.close()

if __name__ == "__main__":
    main()