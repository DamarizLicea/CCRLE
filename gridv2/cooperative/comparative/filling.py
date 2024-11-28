import ast
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax

def read_comparative_q_table(filename):
    comparative_q_table = {}
    with open(filename, 'r') as file:
        next(file) 
        next(file)  
        for line in file:
            parts = line.split("\t")
            state = ast.literal_eval(parts[0].strip())
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
    agent_selection_rewards_true = {'RL': 0, 'Empowerment': 0}
    agent_selection_rewards_false = {'RL': 0, 'Empowerment': 0}

    with open(output2_file, 'w') as file:
        file.write("Estado\tAgente Seleccionado\tRecompensas\n")
        file.write("-" * 50 + "\n")

        for state, values in comparative_q_table.items():
            q_rl_values = np.array(list(values['Q_RL'].values()))*1
            q_emp_values = np.array(list(values['Q_Emp'].values()))*-1
            
            # SOftmax de los valores de Q
            rl_probs = softmax(q_rl_values)
            emp_probs = softmax(q_emp_values)
            
            # Decidir qué agente inhibir basado en las probabilidades y la existencia de recompensas
            if values['Recompensas']:
                # Si hay recompensas, va RL
                selected_agent = 'RL' if np.mean(rl_probs) > np.mean(emp_probs) else 'Empowerment'
                agent_selection_rewards_true[selected_agent] += 1
            else:
                # Si no hay recompensas, usar Empowerment
                selected_agent = 'Empowerment'
                agent_selection_rewards_false[selected_agent] += 1
            
            print(f"Estado: {state}, Agente seleccionado: {selected_agent}")
            agent_selection[selected_agent] += 1
            state_agent_selection[state] = selected_agent
            file.write(f"{state}\t{selected_agent}\t{values['Recompensas']}\n")

    return agent_selection, state_agent_selection, agent_selection_rewards_true, agent_selection_rewards_false

def plot_agent_selection(agent_selection, state_agent_selection, agent_selection_rewards_true, agent_selection_rewards_false):
    # Gráfico
    agents = list(agent_selection.keys())
    counts = list(agent_selection.values())

    plt.figure(figsize=(10, 5))
    plt.bar(agents, counts, color=['blue', 'orange'])
    plt.xlabel('Agente')
    plt.ylabel('Número de selecciones')
    plt.title('Selección total de agentes')
    plt.show()
    # Gráfico de selección de agentes cuando las recompensas son True
    agents_true = list(agent_selection_rewards_true.keys())
    counts_true = list(agent_selection_rewards_true.values())

    plt.figure(figsize=(10, 5))
    plt.bar(agents_true, counts_true, color=['blue', 'orange'])
    plt.xlabel('Agente')
    plt.ylabel('Número de selecciones')
    plt.title('Selección de agentes cuando las recompensas son True')
    plt.show()

    # Gráfico de selección de agentes cuando las recompensas son False
    agents_false = list(agent_selection_rewards_false.keys())
    counts_false = list(agent_selection_rewards_false.values())

    plt.figure(figsize=(10, 5))
    plt.bar(agents_false, counts_false, color=['blue', 'orange'])
    plt.xlabel('Agente')
    plt.ylabel('Número de selecciones')
    plt.title('Selección de agentes cuando las recompensas son False')
    plt.show()


input_file = 'q_comparative_with_rewards.txt'
output_file = 'q_comparative_duplicated.txt'
output2_file = 'q_comparative_softmax.txt'

comparative_q_table = read_comparative_q_table(input_file)

# Filling
duplicated_comparative_q_table = duplicate_states_with_rewards(comparative_q_table)
write_comparative_q_table(output_file, duplicated_comparative_q_table)

# Softmax
agent_selection, state_agent_selection, agent_selection_rewards_true ,agent_selection_rewards_false = select_agent_based_on_softmax(duplicated_comparative_q_table)

plot_agent_selection(agent_selection, state_agent_selection, agent_selection_rewards_true, agent_selection_rewards_false)