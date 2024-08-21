import numpy as np

INITIAL_STONES = 4
PLAYER1_START = 0
PLAYER2_START = 7
BOARD_SIZE = 14

# Constantes de aprendizaje
ALPHA = 0.1  # tasa de aprendizaje
GAMMA = 0.9  # factor de descuento
EPSILON = 0.1  # tasa de exploración

class MancalaGame:
    def __init__(self):
        self.board = np.zeros(BOARD_SIZE, dtype=int)
        self.board[PLAYER1_START+1:PLAYER1_START+7] = INITIAL_STONES
        self.board[PLAYER2_START+1:PLAYER2_START+7] = INITIAL_STONES
        self.player_turn = 0  # 0 para jugador 1, 1 para jugador 2

    def print_board(self):
        print("Estado del tablero:")
        print(self.board)

    def make_move(self, player, action):
        # Determinar el rango de casillas del jugador
        if player == 0:
            start = PLAYER1_START + 1
            end = PLAYER1_START + 6
            deposit = PLAYER1_START
        else:
            start = PLAYER2_START + 1
            end = PLAYER2_START + 6
            deposit = PLAYER2_START

        # Tomar las piedras de la casilla seleccionada
        stones = self.board[start + action - 1]
        self.board[start + action - 1] = 0
        index = start + action

        # Distribuir las piedras
        while stones > 0:
            if index == BOARD_SIZE:
                index = 0
            if (player == 0 and index == PLAYER2_START) or (player == 1 and index == PLAYER1_START):
                index += 1
                continue
            self.board[index] += 1
            stones -= 1
            index += 1

        # Ajustar el índice al último lugar donde se colocó una piedra
        index -= 1
        if index < 0:
            index = BOARD_SIZE - 1

        # Captura de piedras
        if self.board[index] == 1 and start <= index <= end:
            opposite_index = BOARD_SIZE - 2 - index
            if self.board[opposite_index] > 0:
                self.board[deposit] += self.board[opposite_index] + 1
                self.board[index] = 0
                self.board[opposite_index] = 0

        # Repetir turno si la última piedra cae en el depósito del jugador
        if index == deposit:
            return player

        # Cambiar de turno
        return 1 - player

class QLearningAgent:
    def __init__(self):
        self.q_values = {}

    def get_q_value(self, state, action):
        state_tuple = tuple(state)  # Convertir el estado a una tupla
        if (state_tuple, action) not in self.q_values:
            self.q_values[(state_tuple, action)] = 0.0
        return self.q_values[(state_tuple, action)]

    def choose_action(self, state):
        state_tuple = tuple(state)  # Convertir el estado a una tupla
        return max(range(1, 7), key=lambda a: self.get_q_value(state_tuple, a))

def play_game(agent1, agent2, game):
    for turn in range(10):  # Simular 10 turnos
        if game.player_turn == 0:
            action = agent1.choose_action(game.board)
            game.player_turn = game.make_move(0, action)
        else:
            action = agent2.choose_action(game.board)
            game.player_turn = game.make_move(1, action)
        
        game.print_board()

# Ejemplo de uso
game = MancalaGame()
game.print_board()
agent1 = QLearningAgent()
agent2 = QLearningAgent()
play_game(agent1, agent2, game)
