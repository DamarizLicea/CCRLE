import os
import json
from gridv2.inicios.main import SimpleEnv
from minigrid.core.world_object import Goal, Wall
import numpy as np
import matplotlib.pyplot as plt

class EmpowermentAnalysis:
    def __init__(self, env, max_steps=90, cache_file="empowerment_cache.json"):
        self.env = env
        self.max_steps = max_steps
        self.cache_file = cache_file
        self.empowerment_growth = self.load_cache()

    def load_cache(self):
        """Carga los datos de empowerment desde un archivo, si existe."""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as f:
                return {int(k): np.array(v) for k, v in json.load(f).items()}
        return {}

    def save_cache(self):
        """Guarda los datos de empowerment en un archivo."""
        with open(self.cache_file, "w") as f:
            json.dump({k: v.tolist() for k, v in self.empowerment_growth.items()}, f)

    def calculate_empowerment_across_steps(self):
        """Calcula y guarda los valores de empowerment para los pasos faltantes."""
        for step in range(1, self.max_steps + 1):
            if step not in self.empowerment_growth:
                print(f"Calculando empowerment para {step} pasos...")
                self.empowerment_growth[step] = np.full_like(self.env.empowerment_grid, -1, dtype=float)
                
                for row in range(1, self.env.grid.height - 1):
                    for col in range(1, self.env.grid.width - 1):
                        cell = self.env.grid.get(col, row)
                        if not isinstance(cell, Wall):
                            self.empowerment_growth[step][row, col] = self.env.calculate_empowerment(
                                start_x=col, start_y=row, num_pasos=step
                            )
        self.save_cache()

    def plot_empowerment_differences(self):
        """Grafica las diferencias entre valores consecutivos de empowerment."""
        steps = sorted(self.empowerment_growth.keys())
        max_empowerment = [np.max(self.empowerment_growth[step]) for step in steps]

        # Calcular las diferencias entre pasos consecutivos
        differences = [max_empowerment[i] - max_empowerment[i - 1] for i in range(1, len(max_empowerment))]

        plt.figure(figsize=(10, 6))
        plt.plot(steps[1:], differences, marker='o', label="Diferencias de Empowerment")

        plt.xlabel("Número de pasos")
        plt.ylabel("Diferencias de Empowerment")
        plt.title("Diferencias entre los valores de Empowerment a pasos consecutivos")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_empowerment_growth(self):
        """Grafica el crecimiento del empowerment y marca puntos clave."""
        steps = sorted(self.empowerment_growth.keys())
        max_empowerment = [np.max(self.empowerment_growth[step]) for step in steps]

        plt.figure(figsize=(10, 6))
        plt.plot(steps, max_empowerment, marker='o', label="Empowerment máximo")

        plt.xlabel("Número de pasos")
        plt.ylabel("Empowerment máximo")
        plt.title("Crecimiento del Empowerment a través de los pasos")
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == "__main__":
    env = SimpleEnv(render_mode="human")
    env.reset()
    analysis = EmpowermentAnalysis(env, max_steps=10)
    analysis.calculate_empowerment_across_steps()
    analysis.plot_empowerment_growth()
    analysis.plot_empowerment_differences()
