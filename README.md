# Cognitive Cooperative Reinforcement Learning and Empowerment (CCRLE)

**CCRLE** es una colección de proyectos de aprendizaje por refuerzo que implementan el concepto de **Aprendizaje por Refuerzo Cooperativo Cognitivo y Empoderamiento**. El proyecto incluye tres mini-proyectos que demuestran aplicaciones de aprendizaje por refuerzo (RL) y cálculo de empoderamiento en distintos entornos.

## Estructura del Proyecto

- **gridV2**: Implementa un agente en un entorno `EmptyEnv` de MiniGrid. En este proyecto, un agente busca recompensas aleatorias en una cuadrícula usando Q-learning, mientras que un segundo agente calcula el empoderamiento del tablero, permitiendo identificar celdas donde el agente tiene mayor influencia a futuro.
- **taxi**: Utiliza el entorno de taxi de Gymnasium. El agente calcula el empoderamiento en múltiples pasos en el tablero, lo cual mejora sus decisiones mientras transporta pasajeros a sus destinos en el menor número de pasos para maximizar su recompensa.
- **mancala**: Una implementación del juego Mancala en consola. Un agente entrenado con aprendizaje por refuerzo juega contra un usuario humano, tratando de ganar al aplicar estrategias aprendidas.

## Propósito

Este proyecto explora cómo el aprendizaje por refuerzo y el cálculo de empoderamiento pueden combinarse para mejorar la cooperación y optimización de decisiones en diferentes entornos. El **empoderamiento** se utiliza como una métrica para medir la influencia que el agente tiene sobre el entorno, incentivando decisiones que maximicen sus opciones futuras.

