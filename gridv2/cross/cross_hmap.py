import matplotlib.pyplot as plt
import numpy as np

def plot_empowerment_heatmap(empowerment_matrix):
    plt.figure(figsize=(8, 8))
    plt.imshow(empowerment_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label="Empowerment")
    plt.title("Heatmap del Empowerment en Cruz")
    plt.xlabel("Posición X")
    plt.ylabel("Posición Y")
    plt.gca().invert_yaxis() 
    plt.show()


empowerment_matrix = np.loadtxt("empowerment_cross_matrix.txt")

plot_empowerment_heatmap(empowerment_matrix)