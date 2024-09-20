import numpy as np
import csv
import pickle
from collections import defaultdict
from termcolor import colored

# Funciones básicas del juego de Mancala
def inicializar_tablero():
    return [4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0]

def mostrar_tablero(tablero, jugador_actual):
    print("      11   10   9   8   7   6")
    print("   +----+----+----+----+----+----+")
    print(f"   | {colored(tablero[12], 'yellow')} | {colored(tablero[11], 'yellow')} | {colored(tablero[10], 'yellow')} | {colored(tablero[9], 'yellow')} | {colored(tablero[8], 'yellow')} | {colored(tablero[7], 'yellow')} |")
    print("   +----+----+----+----+----+----+")
    print(f" {colored(tablero[13], 'yellow')}  |                    | {colored(tablero[6], 'cyan')}  ")
    print("   +----+----+----+----+----+----+")
    print(f"   | {colored(tablero[0], 'cyan')} | {colored(tablero[1], 'cyan')} | {colored(tablero[2], 'cyan')} | {colored(tablero[3], 'cyan')} | {colored(tablero[4], 'cyan')} | {colored(tablero[5], 'cyan')} |")
    print("   +----+----+----+----+----+----+")
    print("      0    1    2    3    4    5")

    print(colored(f"\nTurno del Jugador {jugador_actual}", 'green'))

def realizar_movimiento(tablero, pos, jugador):
    semillas = tablero[pos]
    tablero[pos] = 0
    i = pos

    while semillas > 0:
        i = (i + 1) % 14
        if (jugador == 1 and i == 13) or (jugador == 2 and i == 6):
            continue
        tablero[i] += 1
        semillas -= 1

    return i

def capturar(tablero, pos, jugador):
    oponente = 12 - pos
    if tablero[pos] == 1 and tablero[oponente] > 0:
        if (jugador == 1 and 0 <= pos <= 5) or (jugador == 2 and 7 <= pos <= 12):
            tablero[jugador * 7 - 1] += tablero[oponente] + 1
            tablero[pos] = 0
            tablero[oponente] = 0

def turno(jugador):
    return 1 if jugador == 2 else 2

def fin_del_juego(tablero):
    return sum(tablero[0:6]) == 0 or sum(tablero[7:13]) == 0

def agregar_semillas_restantes(tablero):
    tablero[6] += sum(tablero[0:6])
    tablero[13] += sum(tablero[7:13])
    for i in range(0, 6):
        tablero[i] = 0
    for i in range(7, 13):
        tablero[i] = 0

# Funciones de Empoderamiento
def calcular_empoderamiento(tablero, jugador):
    posibles_estados = set()
    for accion in range(6):
        copia_tablero = tablero.copy()
        ultima_pos = realizar_movimiento(copia_tablero, accion + (7 if jugador == 2 else 0), jugador)
        capturar(copia_tablero, ultima_pos, jugador)
        posibles_estados.add(tuple(copia_tablero))
    return len(posibles_estados)

# Agente Cooperativo que usa Reinforcement Learning y Cognitive Cooperation
class AgenteCooperativoUnificado:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, initial_exploration_rate=1.0, min_exploration_rate=0.05, decay=0.995):
        self.q_table = defaultdict(float)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = initial_exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.decay = decay
        self.cargar_q_table()

    def obtener_valor_q(self, estado, accion):
        return self.q_table.get((estado, accion), 0.0)

    def actualizar_q_valor(self, estado, accion, recompensa, siguiente_estado):
        maximo_valor_q = max([self.obtener_valor_q(siguiente_estado, a) for a in range(6)], default=0.0)
        valor_q_actual = self.obtener_valor_q(estado, accion)
        nuevo_valor_q = valor_q_actual + self.learning_rate * (recompensa + self.discount_factor * maximo_valor_q - valor_q_actual)
        self.q_table[(estado, accion)] = nuevo_valor_q

    def elegir_accion(self, tablero, jugador):
        estado = tuple(tablero)
        if np.random.rand() < self.exploration_rate:
            return np.random.choice([i for i in range(6) if tablero[i + (7 if jugador == 2 else 0)] > 0])
        else:
            valores_q = [self.obtener_valor_q(estado, a) for a in range(6)]
            empoderamientos = [calcular_empoderamiento(self.simular_tablero(tablero, jugador, a), jugador) for a in range(6)]
            valores_combinados = [v + e for v, e in zip(valores_q, empoderamientos)]
            return np.argmax(valores_combinados)

    def simular_tablero(self, tablero, jugador, accion):
        copia_tablero = tablero.copy()
        ultima_pos = realizar_movimiento(copia_tablero, accion + (7 if jugador == 2 else 0), jugador)
        capturar(copia_tablero, ultima_pos, jugador)
        return copia_tablero

    def guardar_q_table(self):
        with open('q_table_cooperativo.pkl', 'wb') as f:
            pickle.dump(self.q_table, f)

    def cargar_q_table(self):
        try:
            with open('q_table_cooperativo.pkl', 'rb') as f:
                self.q_table = pickle.load(f)
        except FileNotFoundError:
            self.q_table = {}

    def decaimiento_exploracion(self):
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.decay)

# Agente Competitivo que es una instancia del Agente Cooperativo
class AgenteCompetitivo(AgenteCooperativoUnificado):
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.2):
        super().__init__(learning_rate, discount_factor, initial_exploration_rate=exploration_rate)
        self.exploration_rate = exploration_rate

# Función para registrar una partida en un archivo CSV
def registrar_partida(partida_num, ganador, tablero):
    with open('historial_partidas.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([partida_num, ganador, tablero])

def entrenar_agente_cooperativo_y_registrar(partidas=100):
    agente_cooperativo = AgenteCooperativoUnificado()
    
    ganadas = 0
    empates = 0

    with open('historial_partidas.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Partida", "Ganador", "Tablero Final"])

    for partida_num in range(1, partidas + 1):
        tablero = inicializar_tablero()
        jugador = 1

        while not fin_del_juego(tablero):
            if jugador == 1:
                pos = agente_cooperativo.elegir_accion(tablero, jugador)
            else:
                pos = agente_cooperativo.elegir_accion(tablero, jugador) + 7

            ultima_pos = realizar_movimiento(tablero, pos, jugador)
            capturar(tablero, ultima_pos, jugador)

            if ultima_pos != jugador * 7 - 1:
                jugador = turno(jugador)

            estado_actual = tuple(tablero)
            siguiente_estado = tuple(tablero)
            recompensa = tablero[13] - tablero[6]

            agente_cooperativo.actualizar_q_valor(estado_actual, pos - 7 if jugador == 2 else pos, recompensa, siguiente_estado)

        agregar_semillas_restantes(tablero)

        if tablero[6] > tablero[13]:
            ganador = "Agente Cooperativo"
            ganadas += 1
        elif tablero[13] > tablero[6]:
            ganador = "Agente Competitivo"
        else:
            ganador = "Empate"
            empates += 1

        registrar_partida(partida_num, ganador, tablero)
        print(f"Partida {partida_num}: {ganador}")

        agente_cooperativo.decaimiento_exploracion()

    print(f"\nAgente Cooperativo: {ganadas} victorias, {empates} empates")
    agente_cooperativo.guardar_q_table()

if __name__ == "__main__":
    entrenar_agente_cooperativo_y_registrar()