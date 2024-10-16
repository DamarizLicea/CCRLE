import numpy as np

# Funciones básicas del juego
def inicializar_tablero():
    return [4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0]

def mostrar_tablero(tablero):
    print(f"   {tablero[12]} {tablero[11]} {tablero[10]} {tablero[9]} {tablero[8]} {tablero[7]}")
    print(f"{tablero[13]}                   {tablero[6]}")
    print(f"   {tablero[0]} {tablero[1]} {tablero[2]} {tablero[3]} {tablero[4]} {tablero[5]}")

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

    # Captura
    if jugador == 1 and 0 <= i <= 5 and tablero[i] == 1:
        tablero[6] += tablero[12 - i] + 1
        tablero[i] = 0
        tablero[12 - i] = 0
    elif jugador == 2 and 7 <= i <= 12 and tablero[i] == 1:
        tablero[13] += tablero[12 - i] + 1
        tablero[i] = 0
        tablero[12 - i] = 0

    if (jugador == 1 and i == 6) or (jugador == 2 and i == 13):
        return jugador

    return 1 if jugador == 2 else 2

def es_juego_terminado(tablero):
    return sum(tablero[0:6]) == 0 or sum(tablero[7:13]) == 0

def recolectar_semillas_restantes(tablero):
    tablero[6] += sum(tablero[0:6])
    tablero[13] += sum(tablero[7:13])
    for i in range(0, 6):
        tablero[i] = 0
    for i in range(7, 13):
        tablero[i] = 0

# Agente
class AgenteReforzamiento:
    def __init__(self):
        self.q_table = {}

    def elegir_accion(self, estado):
        if estado not in self.q_table:
            self.q_table[estado] = np.zeros(6)
        return np.argmax(self.q_table[estado])

    def actualizar_q_valor(self, estado, accion, recompensa, siguiente_estado):
        if estado not in self.q_table:
            self.q_table[estado] = np.zeros(6)
        if siguiente_estado not in self.q_table:
            self.q_table[siguiente_estado] = np.zeros(6)
        self.q_table[estado][accion] = recompensa + 0.9 * np.max(self.q_table[siguiente_estado])

def jugar():
    tablero = inicializar_tablero()
    jugador = 1
    agente = AgenteReforzamiento()

    while not es_juego_terminado(tablero):
        mostrar_tablero(tablero)
        if jugador == 1:
            pos = int(input(f"Jugador {jugador}, introduce tu movimiento (1-6): ")) - 1
        else:
            estado = tuple(tablero)
            pos = agente.elegir_accion(estado) + 7

        jugador = realizar_movimiento(tablero, pos, jugador)

    recolectar_semillas_restantes(tablero)
    mostrar_tablero(tablero)
    print(f"Partida terminada. Recompensa final: {tablero[6] - tablero[13]}")

jugar()