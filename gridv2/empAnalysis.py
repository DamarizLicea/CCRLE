from main import SimpleEnv, PassableWall  # Importar tu entorno
from minigrid.core.world_object import Goal, Wall
import numpy as np
import matplotlib.pyplot as plt

class EmpowermentAnalysis:
    def __init__(self, env, max_steps=90):
        self.env = env
        self.max_steps = max_steps
        self.empowerment_growth = {}

    def calculate_empowerment_across_steps(self):
        """Función que calcula el empowerment en cada celda 
            accesible para múltiples pasos a futuro, hasta max_steps."""
        self.empowerment_growth = {step: np.full_like(self.env.empowerment_matrix, -1, dtype=float) for step in range(1, self.max_steps + 1)}
        
        for step in range(1, self.max_steps + 1):
            for row in range(1, self.env.grid.height - 1):
                for col in range(1, self.env.grid.width - 1):
                    cell = self.env.grid.get(col, row)
                    if not isinstance(cell, Wall) or isinstance(cell, PassableWall):
                        self.empowerment_growth[step][row, col] = self.env.calculate_empowerment(col, row, n_steps=step)

    def analyze_empowerment_growth(self):
        """ Función que analiza el crecimiento del empowerment para encontrar el punto de 
            saturación y el mayor incremento del empowerment hasta max_steps."""
        max_growth_step = None
        max_saturation_step = None
        max_growth = -np.inf
        threshold = 0.01  # Umbral para detectar falta de crecimiento en empowerment
        consecutive_steps = 0
        saturation_steps_threshold = 5  # Pasos consecutivos sin cambios para marcar punto de saturación

        with open("empowerment_analysis.txt", "w") as file:
            file.write("Paso\tEmpowerment Máximo\n")

            for step in range(1, len(self.empowerment_growth)):
                max_empowerment_step = np.max(self.empowerment_growth[step])
                max_empowerment_next = np.max(self.empowerment_growth[step + 1])
                diff = np.abs(max_empowerment_next - max_empowerment_step)
                
                file.write(f"{step}\t{max_empowerment_step}\n")

                if diff > max_growth:
                    max_growth = diff
                    max_growth_step = step

                if diff < threshold:
                    consecutive_steps += 1
                    if consecutive_steps >= saturation_steps_threshold and max_saturation_step is None:
                        max_saturation_step = step - saturation_steps_threshold + 1
                else:
                    consecutive_steps = 0 

        print(f"El empowerment alcanza la saturación aproximadamente en el paso: {max_saturation_step}")
        print(f"El mayor crecimiento en empowerment ocurre en el paso: {max_growth_step}")
        
        return max_growth_step, max_saturation_step


    def plot_empowerment_growth(self):
        """Función para graficar el crecimiento de empowerment y marca los puntos de interes."""
        max_growth_step, max_saturation_step = self.analyze_empowerment_growth()

        steps = list(self.empowerment_growth.keys())
        max_empowerment = [np.max(self.empowerment_growth[step]) for step in steps]

        plt.figure(figsize=(10, 6))
        plt.plot(steps, max_empowerment, marker='o', label="Empowerment máximo")

        # Punto de mayor crecimiento
        if max_growth_step:
            plt.axvline(x=max_growth_step, color='r', linestyle='--', label=f"Mayor crecimiento en paso {max_growth_step}")

        # Punto de saturación
        if max_saturation_step:
            plt.axvline(x=max_saturation_step, color='g', linestyle='--', label=f"Saturación en paso {max_saturation_step}")

        plt.xlabel("Número de pasos")
        plt.ylabel("Empowerment máximo")
        plt.title("Crecimiento del Empowerment a través de los pasos")
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == "__main__":
    env = SimpleEnv(render_mode="human")
    env.reset()
    analysis = EmpowermentAnalysis(env, max_steps=21)
    analysis.calculate_empowerment_across_steps()
    analysis.plot_empowerment_growth()
