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

def juego_mancala():
    tablero = inicializar_tablero()
    jugador = 1

    while not fin_del_juego(tablero):
        mostrar_tablero(tablero)
        print(f"Turno del Jugador {jugador}")

        while True:
            pos = int(input("Elige una posición (0-5 para Jugador 1, 7-12 para Jugador 2): "))
            if (jugador == 1 and 0 <= pos <= 5 and tablero[pos] > 0) or (jugador == 2 and 7 <= pos <= 12 and tablero[pos] > 0):
                break
            else:
                print("Movimiento inválido. Elige una casilla que tenga semillas.")

        ultima_pos = realizar_movimiento(tablero, pos, jugador)
        capturar(tablero, ultima_pos, jugador)

        if ultima_pos != jugador * 7 - 1:
            jugador = turno(jugador)

    agregar_semillas_restantes(tablero)
    mostrar_tablero(tablero)
    print(f"Juego terminado. Jugador 1: {tablero[6]}, Jugador 2: {tablero[13]}")

juego_mancala()