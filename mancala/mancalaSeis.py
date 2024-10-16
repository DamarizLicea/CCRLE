import numpy as np
import csv
import pickle
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

    """
    Realiza un movimiento en el tablero desde la posición dada.

    Args:
        tablero (list): Estado actual del tablero.
        pos (int): Posición desde donde se recogerán las semillas.
        jugador (int): Número del jugador que está realizando el movimiento.

    Returns:
        int: Última posición donde se dejó una semilla.
    """

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

    """
    Captura las semillas del pozo opuesto si el último movimiento termina 
    en un pozo vacío del jugador.

    Args:
        tablero (list): Estado actual del tablero.
        pos (int): Última posición donde se dejó una semilla.
        jugador (int): Número del jugador que está realizando el movimiento.
    """

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
    """
    Calcula el empoderamiento para un jugador dado. El empoderamiento mide 
    la cantidad de futuros estados posibles desde el estado actual.

    Args:
        tablero (list): Estado actual del tablero.
        jugador (int): Número del jugador para el cual se calcula el empoderamiento.

    Returns:
        int: Número de estados posibles que se pueden alcanzar.
    """
    posibles_estados = set()
    for accion in range(6):
        copia_tablero = tablero.copy()
        ultima_pos = realizar_movimiento(copia_tablero, accion + (7 if jugador == 2 else 0), jugador)
        capturar(copia_tablero, ultima_pos, jugador)
        posibles_estados.add(tuple(copia_tablero))
    return len(posibles_estados)
#agregar mutual information
#en ambientes discretos

#divergencia kl= difernecia entre distribuciones 
#sacar

# Agente Cooperativo que actúa como uno solo
class AgenteCooperativoUnificado:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1, exploration_decay=0.99):
        """
        Inicializa los parámetros del agente.

        Args:
            learning_rate (float): Tasa de aprendizaje para actualizar los valores Q.
            discount_factor (float): Factor de descuento para ponderar recompensas futuras.
            exploration_rate (float): Tasa de exploración inicial.
            exploration_decay (float): Tasa de decaimiento de la exploración.
        """
        self.q_table = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.cargar_q_table()

    def obtener_valor_q(self, estado, accion):
        """
        Recupera el valor Q para una combinación específica de estado y acción.

        Args:
            estado (tuple): Estado actual del tablero como una tupla.
            accion (int): Acción que se quiere evaluar.

        Returns:
            float: Valor Q correspondiente al estado y acción dados.
        """
        return self.q_table.get((estado, accion), 0.0)

    def actualizar_q_valor(self, estado, accion, recompensa, siguiente_estado):
        """
        Actualiza el valor Q basado en la recompensa recibida y el estado futuro.

        Args:
            estado (tuple): Estado actual del tablero.
            accion (int): Acción realizada.
            recompensa (float): Recompensa recibida por la acción.
            siguiente_estado (tuple): Estado del tablero después de realizar la acción.
        """
        maximo_valor_q = max([self.obtener_valor_q(siguiente_estado, a) for a in range(6)], default=0.0)
        valor_q_actual = self.obtener_valor_q(estado, accion)
        nuevo_valor_q = valor_q_actual + self.learning_rate * (recompensa + self.discount_factor * maximo_valor_q - valor_q_actual)
        self.q_table[(estado, accion)] = nuevo_valor_q

    def elegir_accion(self, tablero, jugador):
        """
        Selecciona la mejor acción basada en los valores Q y el empoderamiento,
        considerando tanto la exploración como la explotación.

        Args:
            tablero (list): Estado actual del tablero.
            jugador (int): Número del jugador que va a tomar la acción.

        Returns:
            int: Acción seleccionada (posición del pozo).
        """
        estado = tuple(tablero)
        if np.random.rand() < self.exploration_rate:
            return np.random.choice([i for i in range(6) if tablero[i + (7 if jugador == 2 else 0)] > 0])
        else:
            valores_q = [self.obtener_valor_q(estado, a) for a in range(6)]
            empoderamiento_valores = [calcular_empoderamiento(tablero, jugador) for a in range(6)]
            valores_combinados = [v + e for v, e in zip(valores_q, empoderamiento_valores)]
            return np.argmax(valores_combinados)

    def guardar_q_table(self):
        with open('q_table_cooperativo.pkl', 'wb') as f:
            pickle.dump(self.q_table, f)

    def cargar_q_table(self):
        try:
            with open('q_table_cooperativo.pkl', 'rb') as f:
                self.q_table = pickle.load(f)
        except FileNotFoundError:
            self.q_table = {}

# Nuevo Agente Competitivo
class AgenteCompetitivo(AgenteCooperativoUnificado):
    def elegir_accion(self, tablero, jugador):
        estado = tuple(tablero)
        if np.random.rand() < self.exploration_rate:
            return np.random.choice([i for i in range(6) if tablero[i + (7 if jugador == 2 else 0)] > 0])
        else:
            valores_q = [self.obtener_valor_q(estado, a) for a in range(6)]
            return np.argmax(valores_q)

# Función para registrar una partida en un archivo CSV
def registrar_partida(partida_num, ganador, tablero):
    with open('historial_partidas.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([partida_num, ganador, tablero])

def entrenar_agente_cooperativo_y_registrar(partidas=10000):
    agente_cooperativo_original = AgenteCooperativoUnificado()
    agente_cooperativo_copia = AgenteCooperativoUnificado()
    
    ganadas_original = 0
    ganadas_copia = 0
    empates = 0

    with open('historial_partidas.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Partida", "Ganador", "Tablero Final"])

    for partida_num in range(1, partidas + 1):
        tablero = inicializar_tablero()
        jugador = 1

        while not fin_del_juego(tablero):
            if jugador == 1:
                pos = agente_cooperativo_original.elegir_accion(tablero, jugador)
            else:
                pos = agente_cooperativo_copia.elegir_accion(tablero, jugador) + 7

            ultima_pos = realizar_movimiento(tablero, pos, jugador)
            capturar(tablero, ultima_pos, jugador)

            if ultima_pos != jugador * 7 - 1:
                jugador = turno(jugador)

            estado_actual = tuple(tablero)
            siguiente_estado = tuple(tablero)
            recompensa = tablero[13] - tablero[6]

            if jugador == 1:
                agente_cooperativo_original.actualizar_q_valor(estado_actual, pos, recompensa, siguiente_estado)
            else:
                agente_cooperativo_copia.actualizar_q_valor(estado_actual, pos - 7, recompensa, siguiente_estado)

        agregar_semillas_restantes(tablero)

        if tablero[6] > tablero[13]:
            ganador = "Agente Cooperativo Original"
            ganadas_original += 1
        elif tablero[13] > tablero[6]:
            ganador = "Agente Cooperativo Copia"
            ganadas_copia += 1
        else:
            ganador = "Empate"
            empates += 1

        registrar_partida(partida_num, ganador, tablero)

        agente_cooperativo_original.exploration_rate *= agente_cooperativo_original.exploration_decay
        agente_cooperativo_copia.exploration_rate *= agente_cooperativo_copia.exploration_decay

    agente_cooperativo_original.guardar_q_table()
    agente_cooperativo_copia.guardar_q_table()

    print(f"Total de partidas jugadas: {partidas}")
    print(f"Agente Cooperativo Original ganó: {ganadas_original}")
    print(f"Agente Cooperativo Copia ganó: {ganadas_copia}")
    print(f"Empates: {empates}")

def jugar_contra_agente_cooperativo():
    agente_cooperativo = AgenteCooperativoUnificado()
    agente_cooperativo.cargar_q_table()

    tablero = inicializar_tablero()
    jugador = 1

    while not fin_del_juego(tablero):
        mostrar_tablero(tablero, jugador)

        if jugador == 1:
            pos = int(input("Ingrese la posición donde desea colocar la semilla (0-5): "))
            while pos < 0 or pos > 5 or tablero[pos] == 0:
                print("Posición inválida. Por favor, ingrese una posición válida.")
                pos = int(input("Ingrese la posición donde desea colocar la semilla (0-5): "))
        else:
            pos = agente_cooperativo.elegir_accion(tablero, jugador) + 7

        ultima_pos = realizar_movimiento(tablero, pos, jugador)
        capturar(tablero, ultima_pos, jugador)

        if ultima_pos != jugador * 7 - 1:
            jugador = turno(jugador)

    agregar_semillas_restantes(tablero)

    mostrar_tablero(tablero, jugador)

    if tablero[6] > tablero[13]:
        print(colored("¡Ganaste!", 'cyan'))
    elif tablero[13] > tablero[6]:
        print(colored("Agente cooperativo ganó.", 'amarillo'))
    else:
        print("Empate.")

def main():
    while True:
        print(colored("Menú de inicio:", 'green'))
        print("1. Entrenar al agente cooperativo")
        print("2. Jugar contra el agente cooperativo")
        print("3. Salir")

        opcion = input("Ingrese su opción: ")

        if opcion == "1":
            partidas = int(input("Ingrese el número de partidas para entrenar al agente: "))
            entrenar_agente_cooperativo_y_registrar(partidas)
        elif opcion == "2":
            jugar_contra_agente_cooperativo()
        elif opcion == "3":
            break
        else:
            print(colored("Opción inválida. Por favor, ingrese una opción válida.", 'red'))

if __name__ == "__main__":
    main()