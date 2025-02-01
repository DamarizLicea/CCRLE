# get all the data from the q tables
# cpunt how many times the agent has selected QL and how many times it has selected EMP
# formula de probabilidad: P= p^x * (1-p)^y, donde si es ql es p pero si sale emp es (1-p)-> funcion de verosimilitud
# find the maximum p value: count_ql (aqui puede variar a count_emp) / total de estados (o de datos)

## array con 0 y 1 al azar
## 0 = emp, 1 = ql
import numpy as np
import re
arr = np.random.randint(0, 2, 20)
count_emp = 0
count_ql = 0
for i in range(len(arr)):
    if arr[i] == 0:
        count_emp += 1
    else:
        count_ql += 1
p = count_ql / len(arr)
print(f"Arreglo: {arr}")
print(f"Maximum likelihood QL (1): {p:.2f}")
print(f"Maximum likelihood EMP (0): {1-p:.2f}")


# paso dos, hacer que funcione con las qtables
# formato de las elecciones resumidas
# Estado	Agente Seleccionado	Recompensas
# --------------------------------------------------
# (110, True)	RL	True
# (110, False)	Empowerment	False
# (111, True)	RL	True
# (111, False)	Empowerment	False
# quiero la maximum likelihood de que el agente seleccione RL o Empowerment si hay, o si no hay recompensa, por cada estado

file_path = "../gridv2/cooperative/comparative/q_comparative_softmax.txt"

count_rl_reward = 0
count_emp_reward = 0
count_rl_no_reward = 0
count_emp_no_reward = 0
total_reward = 0
total_no_reward = 0

with open(file_path, "r") as file:
    lines = file.readlines()
    lines = [line.strip() for line in lines if line.strip()]
    lines = lines[2:] 
    print("\n--- Procesando datos ---\n")

    for line in lines:
        parts = line.split("\t")

        if len(parts) != 3:
            print(f"Error de formato en línea: {repr(line)}")  
            continue

        estado, agente, recompensa = parts
        agente = agente.strip().lower()
        recompensa = recompensa.strip().lower()

        print(f"Estado: {estado}, Agente: {agente}, Recompensa: {recompensa}")

        if recompensa == "true":
            total_reward += 1
            if agente == "rl":
                count_rl_reward += 1
            elif agente == "empowerment":
                count_emp_reward += 1
        elif recompensa == "false":
            total_no_reward += 1
            if agente == "rl":
                count_rl_no_reward += 1
            elif agente == "empowerment":
                count_emp_no_reward += 1

# Paso 2: Calcular probabilidades MLE
p_rl_reward = count_rl_reward / total_reward if total_reward > 0 else 0
p_emp_reward = count_emp_reward / total_reward if total_reward > 0 else 0
p_rl_no_reward = count_rl_no_reward / total_no_reward if total_no_reward > 0 else 0
p_emp_no_reward = count_emp_no_reward / total_no_reward if total_no_reward > 0 else 0

# Paso 3: Imprimir resultados
print(f"\nTotal datos procesados: {total_reward + total_no_reward}")
print(f"Maximum likelihood cuando SI hay recompensa:")
print(f"  Count RL: {count_rl_reward}")
print(f"  RL: {p_rl_reward:.2f}")
print(f"  Count Empowerment: {count_emp_reward}")
print(f"  Empowerment: {p_emp_reward:.2f}")

print(f"\nMaximum likelihood cuando NO hay recompensa:")
print(f"  Count RL: {count_rl_no_reward}")
print(f"  RL: {p_rl_no_reward:.2f}")
print(f"  Count Empowerment: {count_emp_no_reward}")
print(f"  Empowerment: {p_emp_no_reward:.2f}")
