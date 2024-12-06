import ast
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax

"""
Este script lee la tabla Q comparativa y calcula la matriz Q ponderada para cada estado.
Luego, selecciona el agente a utilizar en cada estado basado en el softmax de los valores de Q.
Por último, resume cuántas veces se eligió más Q_RL o Q_Emp basado en los valores ponderados, y los 
guarda en un archivo de texto.
"""

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

def calculate_weighted_q_matrix(comparative_q_table, rewards_remaining):
    """
    Calcular la matriz Q ponderada para cada estado en la tabla Q comparativa.
    """
    all_states = []
    weighted_q_values = []
    
    for state, values in comparative_q_table.items():
        q_rl = np.array(list(values['Q_RL'].values()))  
        q_emp = np.array(list(values['Q_Emp'].values())) 

        q_matrix = np.vstack((q_rl, q_emp))  # Shape (2, acciones)
        
        # Transponer Q (acciones, agentes)
        q_transposed = q_matrix.T

        if rewards_remaining:
            weights = np.array([1, 1])
        else:
            weights = np.array([1, 2])

        # Normalizar pesos 
        normalized_weights = softmax(weights)
        weights_matrix = np.array([normalized_weights] * q_transposed.shape[0]) 
        
        if q_transposed.shape != weights_matrix.shape:
            raise ValueError(f"Shapes no compatibles: q_transposed={q_transposed.shape}, weights={weights_matrix.shape}")
        
        # Aplicar los pesos a Q
        weighted_q = q_transposed * weights_matrix

        all_states.append(state)
        weighted_q_values.append(weighted_q)

    return all_states, weighted_q_values



def select_agent_based_on_softmax(comparative_q_table):
    """
    Seleccionar el agente a utilizar en cada estado basado en el softmax de los valores de Q.
    """
    agent_selection = {'RL': 0, 'Empowerment': 0}
    state_agent_selection = {}
    agent_selection_rewards_true = {'RL': 0, 'Empowerment': 0}
    agent_selection_rewards_false = {'RL': 0, 'Empowerment': 0}

    with open(output2_file, 'w') as file:
        file.write("Estado\tAgente Seleccionado\tRecompensas\n")
        file.write("-" * 50 + "\n")

        for state, values in comparative_q_table.items():
            q_rl_values = np.array(list(values['Q_RL'].values()))
            q_emp_values = np.array(list(values['Q_Emp'].values()))
            
            # SOftmax de los valores de Q
            rl_probs = softmax(q_rl_values)
            emp_probs = softmax(q_emp_values)
            
            # Decidir qué agente inhibir basado en las probabilidades y la existencia de recompensas
            if values['Recompensas']:
                # Si hay recompensas, va RL
                selected_agent = 'RL' 
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


output2_file = 'q_comparative_softmax.txt'


latest_comparative_q_table = r'C:\Users\Damarindo\Desktop\Estancia\gridv2\cooperative\q_table_combinada21.txt'

l_comparative_q_table = read_comparative_q_table(latest_comparative_q_table)

states, weighted_qs = calculate_weighted_q_matrix(l_comparative_q_table, rewards_remaining=True)
states2, weighted_qs2 = calculate_weighted_q_matrix(l_comparative_q_table, rewards_remaining=False)

for state, weighted_q in zip(states, weighted_qs):
    print(f"Estado: {state}")
    print("Matriz Q ponderada:")
    print(weighted_q)

agent_selection, state_agent_selection, agent_selection_rewards_true, agent_selection_rewards_false = select_agent_based_on_softmax(l_comparative_q_table)



def summarize_agent_preferences(states, weighted_q_values):
    """
    Resumen de cuántas veces se eligió más Q_RL o Q_Emp basado en los valores ponderados.
    """
    rl_preference_count = 0  
    emp_preference_count = 0 

    for weighted_q in weighted_q_values:
        # Comparar para cada acción en el estado
        for action_values in weighted_q:
            rl_value, emp_value = action_values 
            print(f"Q_RL: {rl_value}, Q_Emp: {emp_value}")
            if rl_value > emp_value:
                rl_preference_count += 1
            else:
                emp_preference_count += 1

    plt.figure(figsize=(8, 5))
    agents = ['Q_RL', 'Q_Emp']
    preferences = [rl_preference_count, emp_preference_count]
    colors = ['skyblue', 'salmon']

    plt.bar(agents, preferences, color=colors)
    plt.xlabel('Agente')
    plt.ylabel('Número de veces favorecido')
    plt.title('Preferencias de agentes (Q_RL vs Q_Emp)')
    plt.tight_layout()
    plt.show()


    print(f"Q_RL ganó {rl_preference_count} veces.")
    print(f"Q_Emp ganó {emp_preference_count} veces.")


summarize_agent_preferences(states, weighted_qs)
summarize_agent_preferences(states2, weighted_qs2)
