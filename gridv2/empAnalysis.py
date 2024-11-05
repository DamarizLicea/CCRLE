from main import SimpleEnv, PassableWall  # Importar tu entorno
from minigrid.core.world_object import Goal, Wall
import numpy as np
import matplotlib.pyplot as plt

class EmpowermentAnalysis:
    def __init__(self, env, max_steps=20):
        self.env = env
        self.max_steps = max_steps
        self.empowerment_growth = {}

    def calculate_empowerment_across_steps(self):
        """Calcula el empowerment en cada celda accesible para múltiples pasos hasta max_steps."""
        self.empowerment_growth = {step: np.full_like(self.env.empowerment_matrix, -1, dtype=float) for step in range(1, self.max_steps + 1)}
        
        for step in range(1, self.max_steps + 1):
            for row in range(1, self.env.grid.height - 1):
                for col in range(1, self.env.grid.width - 1):
                    cell = self.env.grid.get(col, row)
                    if not isinstance(cell, Wall) or isinstance(cell, PassableWall):
                        self.empowerment_growth[step][row, col] = self.env.calculate_empowerment(col, row, n_steps=step)

    def analyze_empowerment_growth(self):
        """Analiza el crecimiento del empowerment para encontrar el punto de saturación y el mayor incremento."""
        max_growth_step = None
        max_saturation_step = None
        max_growth = -np.inf
        threshold = 0.01  # Define el umbral para considerar que el empowerment dejó de crecer significativamente

        for step in range(1, len(self.empowerment_growth)):
            diff = np.abs(self.empowerment_growth[step + 1] - self.empowerment_growth[step])
            avg_diff = np.mean(diff)

            # Detectar el paso con el crecimiento más significativo
            if avg_diff > max_growth:
                max_growth = avg_diff
                max_growth_step = step

            # Detectar el paso donde la diferencia cae por debajo del umbral
            if avg_diff < threshold and max_saturation_step is None:
                max_saturation_step = step

        print(f"El empowerment alcanza la saturación aproximadamente en el paso: {max_saturation_step}")
        print(f"El mayor crecimiento en empowerment ocurre en el paso: {max_growth_step}")
        
        return max_growth_step, max_saturation_step

    def plot_empowerment_growth(self):
        """Grafica el crecimiento de empowerment y marca los puntos de mayor crecimiento y saturación."""
        max_growth_step, max_saturation_step = self.analyze_empowerment_growth()

        steps = list(self.empowerment_growth.keys())
        avg_empowerment = [np.mean(self.empowerment_growth[step]) for step in steps]

        plt.figure(figsize=(10, 6))
        plt.plot(steps, avg_empowerment, marker='o', label="Empowerment promedio")

        # Marcar el punto de mayor crecimiento
        if max_growth_step:
            plt.axvline(x=max_growth_step, color='r', linestyle='--', label=f"Mayor crecimiento en paso {max_growth_step}")

        # Marcar el punto de saturación
        if max_saturation_step:
            plt.axvline(x=max_saturation_step, color='g', linestyle='--', label=f"Saturación en paso {max_saturation_step}")

        plt.xlabel("Número de pasos")
        plt.ylabel("Empowerment promedio")
        plt.title("Crecimiento del Empowerment a través de los pasos")
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == "__main__":
    # Crear el entorno de SimpleEnv
    env = SimpleEnv(render_mode="human")
    env.reset()

    # Crear el análisis de empowerment
    analysis = EmpowermentAnalysis(env, max_steps=100)
    analysis.calculate_empowerment_across_steps()
    analysis.plot_empowerment_growth()
