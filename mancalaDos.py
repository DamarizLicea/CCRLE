import numpy as np
import random

BOARD_SIZE = 7

class MancalaGame:
    def __init__(self):
        self.board = np.zeros((2, BOARD_SIZE), dtype=int)
        self.board[0, 0] = 7
        self.board[1, 0] = 7
        self.current_player = 0

    def reset(self):
        self.board = np.zeros((2, BOARD_SIZE), dtype=int)
        self.board[0, 0] = 7
        self.board[1, 0] = 7
        self.current_player = 0
        return self.board

    def step(self, action):
        if self.current_player == 0:
            pit = action
            stones = self.board[0, pit]
            self.board[0, pit] = 0
            while stones > 0:
                pit = (pit + 1) % BOARD_SIZE
                self.board[0, pit] += 1
                stones -= 1
            if pit == 0:
                self.current_player = 1
            else:
                self.current_player = 0
        else:
            pit = action
            stones = self.board[1, pit]
            self.board[1, pit] = 0
            while stones > 0:
                pit = (pit + 1) % BOARD_SIZE
                self.board[1, pit] += 1
                stones -= 1
            if pit == 0:
                self.current_player = 0
            else:
                self.current_player = 1
        reward = self.get_reward()
        done = self.is_done()
        return self.board, reward, done, {}

    def get_reward(self):
        if self.is_done():
            if np.sum(self.board[0]) > np.sum(self.board[1]):
                return 1
            elif np.sum(self.board[0]) < np.sum(self.board[1]):
                return -1
            else:
                return 0
        else:
            return 0

    def is_done(self):
        return np.sum(self.board[0]) == 0 or np.sum(self.board[1]) == 0

    def get_winner(self):
        if np.sum(self.board[0]) > np.sum(self.board[1]):
            return 0
        elif np.sum(self.board[0]) < np.sum(self.board[1]):
            return 1
        else:
            return -1

class QLearningAgent:
    def __init__(self, alpha, gamma, epsilon):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_values = {}

    def get_q_value(self, state, action):
        state_tuple = tuple(state.tolist())
        if (state_tuple, action) not in self.q_values:
            self.q_values[(state_tuple, action)] = 0
        return self.q_values[(state_tuple, action)]

    def update_q_value(self, state, action, reward, next_state):
        state_tuple = tuple(state.tolist())
        next_state_tuple = tuple(next_state.tolist())
        q_value = self.get_q_value(state, action)
        next_q_value = max([self.get_q_value(next_state, a) for a in range(BOARD_SIZE)])
        self.q_values[(state_tuple, action)] = q_value + self.alpha * (reward + self.gamma * next_q_value - q_value)

    def choose_action(self, state):
        state_tuple = tuple(state.tolist())
        if random.random() < self.epsilon:
            return random.randint(0, BOARD_SIZE - 1)
        else:
            return max(range(BOARD_SIZE), key=lambda a: self.get_q_value(state, a))

def play_game(agent1, agent2, game):
    state = game.reset()
    while True:
        action1 = agent1.choose_action(state)
        next_state, reward, done, _ = game.step(action1)
        agent1.update_q_value(state, action1, reward, next_state)
        state = next_state
        if done:
            break
        action2 = agent2.choose_action(state)
        next_state, reward, done, _ = game.step(action2)
        agent2.update_q_value(state, action2, reward, next_state)
        state = next_state
        if done:
            break
    return game.get_winner()

def main():
    game = MancalaGame()  # Asumo que tienes una clase MancalaGame que representa el juego
    agent1 = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.1)
    agent2 = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.1)
    winner = play_game(agent1, agent2, game)
    print("Winner:", winner)

if __name__ == "__main__":
    main()