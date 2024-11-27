import ast
import numpy as np
import matplotlib.pyplot as plt

def read_comparative_q_table(filename):
    comparative_q_table = {}
    with open(filename, 'r') as file:
        next(file) 
        next(file)  
        for line in file:
            parts = line.split("\t")
            state = int(parts[0])
            q_rl = ast.literal_eval(parts[1].strip())
            q_emp = ast.literal_eval(parts[2].strip())
            recompensas = parts[3].strip() == 'True'
            comparative_q_table[state] = {'Q_RL': q_rl, 'Q_Emp': q_emp, 'Recompensas': recompensas}
    return comparative_q_table

def write_comparative_q_table(filename, comparative_q_table):
    with open(filename, 'w') as file:
        file.write("Estado\tQ_RL\tQ_Emp\tRecompensas\n")
        file.write("-" * 100 + "\n")
        for state, values in comparative_q_table.items():
            file.write(f"{state}\t{values['Q_RL']}\t{values['Q_Emp']}\t{values['Recompensas']}\n")

def duplicate_states_with_rewards(comparative_q_table):
    new_comparative_q_table = {}
    for state, values in comparative_q_table.items():
        # Recompensas como True
        new_comparative_q_table[(state, True)] = {
            'Q_RL': values['Q_RL'],
            'Q_Emp': values['Q_Emp'],
            'Recompensas': True
        }
        # Recompensas como False
        new_comparative_q_table[(state, False)] = {
            'Q_RL': values['Q_RL'],
            'Q_Emp': values['Q_Emp'],
            'Recompensas': False
        }
    return new_comparative_q_table

def softmax(x, temperature=1.0):
    e_x = np.exp((x - np.max(x)) / temperature)
    return e_x / e_x.sum()

def select_agent_based_on_softmax(comparative_q_table):
    agent_selection = {'RL': 0, 'Empowerment': 0}
    state_agent_selection = {}

    for state, values in comparative_q_table.items():
        q_rl_values = np.array(list(values['Q_RL'].values()))
        q_emp_values = np.array(list(values['Q_Emp'].values()))
        
        # SOftmax de los valores de Q
        rl_probs = softmax(q_rl_values)
        emp_probs = softmax(q_emp_values)
        
        # Decidir qué agente inhibir basado en las probabilidades y la existencia de recompensas
        if values['Recompensas']:
            # Si hay recompensas, va RL
            selected_agent = 'RL' if np.mean(rl_probs) > np.mean(emp_probs) else 'Empowerment'
        else:
            # Si no hay recompensas, usar Empowerment
            selected_agent = 'Empowerment'
        
        print(f"Estado: {state}, Agente seleccionado: {selected_agent}")
        agent_selection[selected_agent] += 1
        state_agent_selection[state] = selected_agent

    return agent_selection, state_agent_selection

def plot_agent_selection(agent_selection, state_agent_selection):
    # Gráfico
    agents = list(agent_selection.keys())
    counts = list(agent_selection.values())

    plt.figure(figsize=(10, 5))
    plt.bar(agents, counts, color=['blue', 'orange'])
    plt.xlabel('Agente')
    plt.ylabel('Número de selecciones')
    plt.title('Selección total de agentes')
    plt.show()


input_file = 'q_comparative_with_rewards.txt'
output_file = 'q_comparative_duplicated.txt'

comparative_q_table = read_comparative_q_table(input_file)

# Filling
duplicated_comparative_q_table = duplicate_states_with_rewards(comparative_q_table)
write_comparative_q_table(output_file, duplicated_comparative_q_table)

# Softmax
agent_selection, state_agent_selection = select_agent_based_on_softmax(duplicated_comparative_q_table)

plot_agent_selection(agent_selection, state_agent_selection)