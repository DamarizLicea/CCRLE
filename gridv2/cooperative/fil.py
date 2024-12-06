"""
Este archivo combina las Q-tables de los agentes RL y EmpQL en un solo archivo de texto.
No es necesario ejecutar este script, ya que el archivo 'q_table_combinada21.txt' ya está incluido en el repositorio.
Solo se ejecuta si cambian las Q-tables de los agentes RL y EmpQL.
"""

def parse_qtable(file_path):
    """
    Convierte la Q-table de formato texto plano a un diccionario de Python.
    """
    q_table = {}
    with open(file_path, 'r') as file:
        for line in file:
            state, values = line.strip().split(':', 1)
            state = int(state.replace('State ', '').strip()) 
            q_values = eval(values.strip()) 
            q_table[state] = q_values
    return q_table


q_table_rl = parse_qtable(r'C:\Users\Damarindo\Desktop\Estancia\gridv2\inicios\q_table.txt')
q_table_emp = parse_qtable(r'C:\Users\Damarindo\Desktop\Estancia\gridv2\inicios\q_table_empql2.txt')
q_table_combinada = []

for state in q_table_rl.keys():
    rl_values = q_table_rl[state]
    emp_values = q_table_emp.get(state, {a: 0 for a in rl_values.keys()})
    
    q_table_combinada.append((state, True, rl_values, emp_values))
    q_table_combinada.append((state, False, rl_values, emp_values))

with open('q_table_combinada21.txt', 'w') as file:
    file.write("Estado\tQ_RL\tQ_Emp\tRecompensas\n")
    file.write("-" * 100 + "\n")
    for state, recompensas, q_rl, q_emp in q_table_combinada:
        file.write(f"({state}, {recompensas})\t{q_rl}\t{q_emp}\t{recompensas}\n")

print("Q-table combinada guardada exitosamente en 'q_table_combinada21.txt'.")
