import numpy as np

INITIAL_STONES = 4
PLAYER1_START = 0
PLAYER2_START = 7
BOARD_SIZE = 14

# Constantes de aprendizaje
ALPHA = 0.1  # tasa de aprendizaje
GAMMA = 0.9  # factor de descuento
EPSILON = 0.5  # tasa de exploración aumentada para más exploración

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
            print(f"Jugador {player + 1} repite turno.")
            return player

        # Imprimir el estado del tablero después de cada movimiento
        print(f"Jugador {player + 1} movió. Estado del tablero:")
        self.print_board()

        # Cambiar de turno
        return 1 - player

    def is_game_over(self):
        # Verificar si todas las casillas de un lado están vacías
        player1_side_empty = np.all(self.board[PLAYER1_START+1:PLAYER1_START+7] == 0)
        player2_side_empty = np.all(self.board[PLAYER2_START+1:PLAYER2_START+7] == 0)
        return player1_side_empty or player2_side_empty

    def collect_remaining_stones(self):
        # Mover las piedras restantes al depósito correspondiente
        self.board[PLAYER1_START] += np.sum(self.board[PLAYER1_START+1:PLAYER1_START+7])
        self.board[PLAYER2_START] += np.sum(self.board[PLAYER2_START+1:PLAYER2_START+7])
        self.board[PLAYER1_START+1:PLAYER1_START+7] = 0
        self.board[PLAYER2_START+1:PLAYER2_START+7] = 0

class QLearningAgent:
    def __init__(self):
        self.q_table = {}

    def get_state_key(self, state):
        return str(state)

    def get_q_value(self, state, action):
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(6)  # Suponiendo 6 acciones posibles
        return self.q_table[state_key][action]

    def choose_action(self, state):
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(6)  # Suponiendo 6 acciones posibles
        if np.random.rand() < EPSILON:
            return np.random.randint(1, 7)  # Acción aleatoria
        else:
            return np.argmax(self.q_table[state_key]) + 1  # Mejor acción

    def update_q_value(self, state, action, reward, next_state):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        # Inicializar el estado actual si no existe
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(6)  # Suponiendo 6 acciones posibles

        # Inicializar el siguiente estado si no existe
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(6)  # Suponiendo 6 acciones posibles

        # Actualizar el valor Q
        best_next_action = np.argmax(self.q_table[next_state_key])
        td_target = reward + GAMMA * self.q_table[next_state_key][best_next_action]
        td_error = td_target - self.q_table[state_key][action - 1]
        self.q_table[state_key][action - 1] += ALPHA * td_error

def play_game(agent1, agent2, game):
    step_count = 0  # Contador de pasos
    while not game.is_game_over():  # Continuar hasta que se acaben las fichas de un lado
        state = game.board.copy()
        if game.player_turn == 0:
            action = agent1.choose_action(state)
            game.player_turn = game.make_move(0, action)
        else:
            action = agent2.choose_action(state)
            game.player_turn = game.make_move(1, action)
        
        next_state = game.board.copy()
        
        # Definir la recompensa según las reglas del juego
        reward = calculate_reward(game, game.player_turn)
        
        agent1.update_q_value(state, action, reward, next_state)
        agent2.update_q_value(state, action, reward, next_state)
        
        game.print_board()
        step_count += 1  # Incrementar el contador de pasos

    # Recolectar las piedras restantes
    game.collect_remaining_stones()
    game.print_board()

    # Calcular la recompensa final
    final_reward = calculate_final_reward(game)
    print(f"Partida terminada. Recompensa final: {final_reward}")
    print(f"Total de pasos: {step_count}\n")
    return final_reward

def calculate_reward(game, player_turn):
    # Ejemplo de lógica de recompensa durante el juego
    if player_turn == 0:
        return game.board[PLAYER1_START] - game.board[PLAYER2_START]
    else:
        return game.board[PLAYER2_START] - game.board[PLAYER1_START]

def calculate_final_reward(game):
    # Lógica de recompensa final basada en las fichas en los depósitos
    return game.board[PLAYER1_START] - game.board[PLAYER2_START]

def run_multiple_games(num_games):
    rewards = []
    agent1 = QLearningAgent()
    agent2 = QLearningAgent()
    for i in range(num_games):
        game = MancalaGame()
        print(f"Juego {i + 1}:")
        reward = play_game(agent1, agent2, game)
        rewards.append(reward)
    print(f"Recompensas obtenidas en {num_games} juegos: {rewards}")

# Ejecutar múltiples juegos
run_multiple_games(10)