def parse_qtable(file_path):
    """
    Convierte una Q-table en formato texto plano a un diccionario de Python.
    """
    q_table = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Separar el estado y los valores Q
            state, values = line.strip().split(':', 1)
            state = int(state.replace('State ', '').strip())  # Convertir "State X" a número
            # Convertir los valores Q de texto a diccionario
            q_values = eval(values.strip())  # eval convierte el texto a un diccionario
            q_table[state] = q_values
    return q_table


# Cargar las Q-tables desde los archivos
q_table_rl = parse_qtable(r'C:\Users\Damarindo\Desktop\Estancia\gridv2\inicios\q_table.txt')
q_table_emp = parse_qtable(r'C:\Users\Damarindo\Desktop\Estancia\gridv2\inicios\q_table_empql2.txt')
# Crear la tabla combinada
q_table_combinada = []

for state in q_table_rl.keys():
    rl_values = q_table_rl[state]
    emp_values = q_table_emp.get(state, {a: 0 for a in rl_values.keys()})
    
    # Crear dos versiones: con recompensas True y False
    q_table_combinada.append((state, True, rl_values, emp_values))
    q_table_combinada.append((state, False, rl_values, emp_values))

# Guardar la tabla combinada en un nuevo archivo con formato tabular
with open('q_table_combinada21.txt', 'w') as file:
    # Encabezado
    file.write("Estado\tQ_RL\tQ_Emp\tRecompensas\n")
    file.write("-" * 100 + "\n")
    # Contenido
    for state, recompensas, q_rl, q_emp in q_table_combinada:
        file.write(f"({state}, {recompensas})\t{q_rl}\t{q_emp}\t{recompensas}\n")

print("Q-table combinada guardada exitosamente en 'q_table_combinada21.txt'.")
