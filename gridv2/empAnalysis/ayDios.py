import os
import json
from main import SimpleEnv
from minigrid.core.world_object import Wall
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import zoom


class EmpowermentAnalysis:
    def __init__(self, env, max_steps=50, cache_file="empowerment_cache.json", reduction_factor=2):
        self.env = env
        self.max_steps = max_steps
        self.cache_file = cache_file
        self.reduction_factor = reduction_factor
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

    def downsample_grid(self, grid):
        """Reduce la resolución del grid."""
        return grid[::self.reduction_factor, ::self.reduction_factor]

    def upsample_grid(self, grid, original_shape):
        """Restaura la resolución original del grid."""
        return zoom(grid, (original_shape[0] / grid.shape[0], original_shape[1] / grid.shape[1]))

    def calculate_empowerment_across_steps(self):
        """Calcula y guarda los valores de empowerment para los pasos faltantes."""
        reduced_grid = self.downsample_grid(self.env.empowerment_grid)
        original_shape = self.env.empowerment_grid.shape

        for step in range(1, self.max_steps + 1):
            if step not in self.empowerment_growth:
                print(f"Calculando empowerment para {step} pasos en resolución reducida...")
                reduced_emp_grid = np.full_like(reduced_grid, -1, dtype=float)

                for row in range(1, reduced_grid.shape[0] - 1):
                    for col in range(1, reduced_grid.shape[1] - 1):
                        cell = self.env.grid.get(col * self.reduction_factor, row * self.reduction_factor)
                        if not isinstance(cell, Wall):
                            reduced_emp_grid[row, col] = self.env.calculate_empowerment(start_x=row, start_y=col)
                self.empowerment_growth[step] = self.upsample_grid(reduced_emp_grid, original_shape)

        self.save_cache()

    def interpolate_empowerment(self):
        """Interpola los valores máximos de empowerment para pasos intermedios."""
        steps = sorted(self.empowerment_growth.keys())
        max_empowerment = [np.max(self.empowerment_growth[step]) for step in steps]

        # Crear función de interpolación
        interpolation_function = interp1d(steps, max_empowerment, kind='linear', fill_value="extrapolate")

        # Proyectar para todos los pasos
        projected_steps = list(range(1, self.max_steps + 1))
        projected_empowerment = interpolation_function(projected_steps)

        return projected_steps, projected_empowerment
    
    def find_saturation_point(self, tolerance=1e-5, min_steps=5):
        """
        Encuentra el paso de saturación donde el empowerment deja de crecer significativamente
        por al menos `min_steps` consecutivos.
            tolerance (float): El umbral bajo el cual se considera que el crecimiento es insignificante.
            min_steps (int): El número mínimo de pasos consecutivos necesarios para considerar saturación.
            int: El número de pasos en el que se alcanza la saturación.
        """
        steps, projected_empowerment = self.interpolate_empowerment()
        differences = np.diff(projected_empowerment)

        # Verifica si las diferencias son menores que la tolerancia durante `min_steps` consecutivos
        for i in range(len(differences) - min_steps + 1):
            if np.all(np.abs(differences[i:i + min_steps]) < tolerance):
                return steps[i]  # Retorna el paso inicial de la saturación

        return None  # Retorna None si no se encontró saturación

    def find_step_with_max_growth(self):
        """
        Encuentra el paso con el mayor crecimiento de empowerment entre pasos consecutivos.
            tuple: (paso, crecimiento máximo)
        """
        steps, projected_empowerment = self.interpolate_empowerment()
        differences = np.diff(projected_empowerment)

        max_growth = np.max(differences)
        step_with_max_growth = steps[np.argmax(differences) + 1]

        return step_with_max_growth, max_growth

    def plot_empowerment_growth(self, saturation_point=None, max_growth_step=None):
        """
        Grafica el crecimiento del empowerment interpolado y añade líneas verticales
        para señalar el punto de saturación y el de mayor crecimiento.
        """
        steps, projected_empowerment = self.interpolate_empowerment()

        plt.figure(figsize=(10, 6))
        plt.plot(steps, projected_empowerment, marker='o', label="Empowerment máximo (Interpolado)")

        # Añadir línea vertical para el punto de saturación
        if saturation_point:
            plt.axvline(
                x=saturation_point, 
                color='red', 
                linestyle='--', 
                label=f"Punto de Saturación ({saturation_point})"
            )

        # Añadir línea vertical para el punto de mayor crecimiento
        if max_growth_step:
            plt.axvline(
                x=max_growth_step, 
                color='blue', 
                linestyle='--', 
                label=f"Mayor Crecimiento ({max_growth_step})"
            )

        # Configuración de la gráfica
        plt.xlabel("Número de pasos")
        plt.ylabel("Empowerment máximo")
        plt.title("Crecimiento del Empowerment a través de los pasos (Interpolado)")
        plt.legend()
        plt.grid()
        plt.show()
    def plot_empowerment_differences(self):
        """Grafica las diferencias entre valores consecutivos de empowerment interpolado."""
        steps, projected_empowerment = self.interpolate_empowerment()

        # Calcular las diferencias entre pasos consecutivos
        differences = [projected_empowerment[i] - projected_empowerment[i - 1] for i in range(1, len(projected_empowerment))]

        plt.figure(figsize=(10, 6))
        plt.plot(steps[1:], differences, marker='o', label="Diferencias de Empowerment (Interpolado)")

        plt.xlabel("Número de pasos")
        plt.ylabel("Diferencias de Empowerment")
        plt.title("Diferencias entre los valores de Empowerment a pasos consecutivos (Interpolado)")
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == "__main__":
    env = SimpleEnv(render_mode="human")
    env.reset()
    analysis = EmpowermentAnalysis(env, max_steps=50, reduction_factor=2)
    analysis.calculate_empowerment_across_steps()
    
    # Encuentra el punto de saturación
    saturation_point = analysis.find_saturation_point()
    if saturation_point:
        print(f"El punto de saturación se alcanza en el paso {saturation_point}.")
    else:
        print("No se encontró un punto de saturación dentro de los pasos calculados.")
    
    # Encuentra el paso con el mayor crecimiento
    max_growth_step, max_growth = analysis.find_step_with_max_growth()
    print(f"El mayor crecimiento se da en el paso {max_growth_step}, con un incremento de {max_growth:.4f}.")
    
    # Graficar los resultados
    analysis.plot_empowerment_growth(saturation_point=saturation_point, max_growth_step=max_growth_step)

    analysis.plot_empowerment_differences()
