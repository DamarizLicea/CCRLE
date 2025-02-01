import ast
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
from datetime import datetime

"""
Resultados de la ejecucion de este script:
    Revisar registro de selección de agentes .txt
    Revisar selección de agentes .txt
    Falta el grafico, pero se necesita ejecutar el script en un entorno de python

"""

def read_comparative_q_table(filename):
    # Leer la qtable combinada 21.txt
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

def select_agent_exploration(comparative_q_table, params, epsilon=0.1, log_file=None):
    # Seleccionar agente en cada estado basado en el softmax de los valores de Q
    agent_selection = {'RL': 0, 'Empowerment': 0}
    state_agent_selection = {}
    
    # Escribir encabezado de la sección para el documento de registro
    if log_file:
        log_file.write("\n" + "="*80 + "\n")
        log_file.write(f"Parámetros actuales: {params}\n")
        log_file.write(f"Probabilidades base (softmax): {softmax(params)}\n")
        log_file.write("="*80 + "\n\n")
    
    for state, values in comparative_q_table.items():
        # Ajustar las probabilidades base según la presencia de recompensas
        base_probs = softmax(params)
        
        # Si hay recompensas, aumentar la probabilidad de RL
        if values['Recompensas']:
            adjusted_probs = [base_probs[0] * 1.2, base_probs[1] * 0.8]
        else:
            adjusted_probs = [base_probs[0] * 0.8, base_probs[1] * 1.2]
        
        # Normalizar para asegurar que suman 1
        adjusted_probs = adjusted_probs / np.sum(adjusted_probs)

        if np.random.rand() < epsilon:
            selected_agent = np.random.choice(['RL', 'Empowerment'])
            selection_method = "Exploración"
        else:
            a = np.random.choice([0, 1], p=adjusted_probs)
            selected_agent = 'RL' if a == 0 else 'Empowerment'
            selection_method = "Explotación"
        
        agent_selection[selected_agent] += 1
        state_agent_selection[state] = selected_agent
        
        # Registrar en el archivo
        if log_file:
            log_file.write(f"Estado: {state}\n")
            log_file.write(f"Recompensas presentes: {values['Recompensas']}\n")
            log_file.write(f"Probabilidades ajustadas: RL={adjusted_probs[0]:.4f}, Emp={adjusted_probs[1]:.4f}\n")
            log_file.write(f"Método de selección: {selection_method}\n")
            log_file.write(f"Agente seleccionado: {selected_agent}\n")
            log_file.write("-"*50 + "\n")
    
    return agent_selection, state_agent_selection

def update_agent_probabilities(comparative_q_table, params, lr=0.01):
    r_prom = 0
    # Actualizar las probabilidades de los agentes:
    # params += lr * (grad - action_probs) * (reward - r_prom)

    
    for state, values in comparative_q_table.items():
        q_rl_values = np.array(list(values['Q_RL'].values()))
        q_emp_values = np.array(list(values['Q_Emp'].values()))
        
        action_probs = softmax(params)
        a = np.random.choice([0, 1], p=action_probs)
        
        base_reward = np.mean(q_rl_values) if a == 0 else np.mean(q_emp_values)
        
        if values['Recompensas']:
            reward = base_reward * 1.2 if a == 0 else base_reward * 0.8
        else:
            reward = base_reward * 0.8 if a == 0 else base_reward * 1.2
        
        r_prom += lr * (reward - r_prom)
        
        grad = np.zeros_like(params)
        grad[a] = 1
        
        params += lr * (grad - action_probs) * (reward - r_prom)
    
    return params

if __name__ == "__main__":
    latest_comparative_q_table = r'C:\Users\Damarindo\Desktop\Estancia\gridv2\cooperative\q_table_combinada21.txt'  # Ajusta la ruta
    l_comparative_q_table = read_comparative_q_table(latest_comparative_q_table)

    # Crear archivo de registro
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'registro_seleccion_agentes_{timestamp}.txt'

    params = np.zeros(2)
    lr = 0.01
    n_episodes = 1000

    # Arrays para almacenar datos de la evolución
    rl_probs = []
    emp_probs = []
    episodes = []

    # Abrir archivo de registro
    with open(log_filename, 'w', encoding='utf-8') as log_file:
        log_file.write("REGISTRO DE SELECCIÓN DE AGENTES\n")
        log_file.write(f"Fecha y hora de inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Parámetros iniciales: lr={lr}, n_episodes={n_episodes}\n\n")

        # Entrenamiento
        for episode in range(n_episodes):
            if episode % 100 == 0:  # Registrar cada 100 episodios, acuerdate que estas corriendo 100
                log_file.write(f"\nEPISODIO {episode}\n")
                agent_selection, _ = select_agent_exploration(l_comparative_q_table, params, 
                                                           epsilon=0.1, log_file=log_file)
                
                action_probs = softmax(params)
                rl_probs.append(action_probs[0])
                emp_probs.append(action_probs[1])
                episodes.append(episode)
                
                log_file.write(f"\nResumen del episodio {episode}:\n")
                log_file.write(f"Total selecciones: {agent_selection}\n")
                log_file.write("="*80 + "\n")
            else:
                agent_selection, _ = select_agent_exploration(l_comparative_q_table, params, epsilon=0.1)
            
            params = update_agent_probabilities(l_comparative_q_table, params, lr=lr)

        # Registrar resultados finales
        log_file.write("\nRESULTADOS FINALES\n")
        log_file.write(f"Parámetros finales: {params}\n")
        final_probs = softmax(params)
        log_file.write(f"Probabilidades finales: RL={final_probs[0]:.4f}, Emp={final_probs[1]:.4f}\n")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(episodes, rl_probs, 'b-', label='RL')
    plt.plot(episodes, emp_probs, 'r-', label='Empowerment')
    plt.title('Evolución de Probabilidades de Selección')
    plt.xlabel('Episodios')
    plt.ylabel('Probabilidad')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(episodes, [params[0]]*len(episodes), 'b--', label='Param RL')
    plt.plot(episodes, [params[1]]*len(episodes), 'r--', label='Param Emp')
    plt.title('Evolución de Parámetros')
    plt.xlabel('Episodios')
    plt.ylabel('Valor del Parámetro')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()