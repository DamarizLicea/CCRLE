# Cálculo del Empowerment
# def calculate_empowerment(state, transition_counts, n=1):
#     """
#     Calcula el empoderamiento como la capacidad del canal de la secuencia de acciones y los estados futuros.
    
#     :param state: El estado actual del agente.
#     :param transition_counts: Un arreglo de NumPy que contiene los conteos de transiciones.
#     :param n: Número de pasos a considerar (n-step empowerment).
#     :return: El empoderamiento.
#     """
#     empowerment = 0.0
#     for action_seq in range(transition_counts.shape[1]):
#         total_transitions_action = np.sum(transition_counts[state, action_seq])
        
#         # Si no hay transiciones válidas para esta acción, la ignoramos
#         if total_transitions_action == 0:
#             continue

#         # Calcular la probabilidad de transición para cada acción
#         prob_state_given_action = transition_counts[state, action_seq] / total_transitions_action
        
#         for future_state in range(transition_counts.shape[2]):
#             total_transitions_state = np.sum(transition_counts[future_state])
            
#             # Si el estado futuro no tiene transiciones válidas, lo ignoramos
#             if total_transitions_state == 0:
#                 continue

#             prob_state_action = transition_counts[state, action_seq, future_state] / total_transitions_state
            
#             # Si la probabilidad de llegar a un estado futuro es mayor a cero, usamos esa contribución
#             if prob_state_given_action[future_state] > 0 and prob_state_action > 0:
#                 # Aplicar la fórmula de información mutua para calcular la contribución al empowerment
#                 empowerment += prob_state_given_action[future_state] * np.log2(prob_state_given_action[future_state] / prob_state_action)
    
#     return empowerment