# Proyecto Q-Learning con Webots

Este proyecto implementa el algoritmo de **Q-Learning** para entrenar un robot E-puck a navegar autónomamente desde una posición inicial hasta un objetivo (caja verde) evitando obstáculos en un entorno simulado con Webots.

## Tabla de Contenidos

- [Descripción General](#descripción-general)
- [¿Qué es Q-Learning?](#qué-es-q-learning)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Requisitos](#requisitos)
- [Cómo Ejecutar](#cómo-ejecutar)
- [Componentes del Sistema](#componentes-del-sistema)
- [Sistema de Recompensas](#sistema-de-recompensas)
- [Parámetros de Aprendizaje](#parámetros-de-aprendizaje)
- [Observando el Aprendizaje](#observando-el-aprendizaje)
- [Personalización](#personalización)
- [Solución de Problemas](#solución-de-problemas)

---

## Descripción General

Este proyecto demuestra cómo un robot puede **aprender por sí mismo** a navegar en un entorno desconocido usando aprendizaje por refuerzo. El robot no tiene instrucciones preprogramadas sobre cómo llegar al objetivo; en su lugar, aprende mediante prueba y error, recibiendo recompensas y penalizaciones según sus acciones.

### Características Principales

- **Aprendizaje Autónomo**: El robot aprende sin programación explícita de rutas
- **Detección Visual**: Usa su cámara para identificar el objetivo verde
- **Evitación de Obstáculos**: Aprende a evitar paredes usando sensores de distancia
- **Persistencia**: La tabla Q se guarda automáticamente para continuar el aprendizaje
- **Múltiples Episodios**: Entrenamiento continuo con reinicio automático

---

## ¿Qué es Q-Learning?

**Q-Learning** es un algoritmo de aprendizaje por refuerzo que permite a un agente (robot) aprender la mejor acción a tomar en cada situación (estado).

### Conceptos Clave

1. **Estado (State)**: Representación de la situación actual del robot
   - En este proyecto: Lecturas discretizadas de 4 grupos de sensores (frente, izquierda, derecha, atrás)

2. **Acción (Action)**: Lo que el robot puede hacer
   - Avanzar
   - Girar a la izquierda
   - Girar a la derecha

3. **Recompensa (Reward)**: Feedback sobre qué tan buena fue una acción
   - Positiva: Llegar al objetivo, ver el objetivo
   - Negativa: Chocar con obstáculos

4. **Tabla Q**: Almacena el valor de cada par (estado, acción)
   - El robot consulta esta tabla para decidir qué hacer

### Ecuación de Q-Learning

```
Q(s,a) = Q(s,a) + α * (r + γ * max(Q(s',a')) - Q(s,a))
```

Donde:
- **α (alpha)**: Tasa de aprendizaje (0.1) - qué tan rápido aprende
- **γ (gamma)**: Factor de descuento (0.9) - importancia de recompensas futuras
- **r**: Recompensa recibida
- **s**: Estado actual
- **s'**: Siguiente estado
- **a**: Acción tomada

---

## Estructura del Proyecto

```
q-learning/
│
├── worlds/
│   └── proyecto-q-learning.wbt    # Mundo de Webots con robot, arena y objetivo
│
├── controllers/
│   └── q_learning_controller/
│       └── q_learning_controller.py    # Controlador con algoritmo Q-Learning
│
├── q_table.pkl                    # Tabla Q guardada (se crea automáticamente)
├── instrucciones.txt              # Instrucciones originales del proyecto
└── README.md                      # Este archivo
```

---

## Requisitos

- **Webots R2025a** o superior
- **Python 3.8+** (incluido con Webots)
- Librerías (incluidas en Webots):
  - `numpy`
  - `pickle`
  - `random`

---

## Cómo Ejecutar

### Primera Ejecución

1. Abre **Webots**
2. Ve a `File > Open World`
3. Selecciona: `D:\proyectos_webots\q-learning\worlds\proyecto-q-learning.wbt`
4. Presiona el botón **Play** (▶️)
5. Observa la consola para ver el progreso del entrenamiento

### Ejecuciones Posteriores

La tabla Q se guarda automáticamente cada 10 episodios en `q_table.pkl`. Si ejecutas de nuevo, el robot continuará aprendiendo desde donde quedó.

Para **empezar desde cero**, elimina el archivo `q_table.pkl`.

---

## Componentes del Sistema

### 1. Entorno (Mundo de Webots)

**Elementos:**
- **Arena**: Espacio de 2x2 metros con paredes perimetrales
- **Robot E-puck**:
  - Posición inicial: (-0.7, -0.7)
  - 8 sensores de distancia infrarroja
  - Cámara de 64x48 píxeles
  - 2 motores (ruedas izquierda y derecha)
- **Objetivo**: Caja verde en posición (0.7, 0.7)
- **Obstáculos**: 2 paredes internas

### 2. Sistema de Percepción

**Sensores de Distancia:**
- 8 sensores distribuidos alrededor del robot
- Agrupados en 4 direcciones: frente, izquierda, derecha, atrás
- Cada dirección se discretiza en 3 niveles:
  - `0` = Libre (distancia > 15%)
  - `1` = Cerca (15% < distancia < 40%)
  - `2` = Muy cerca (distancia > 40%)

**Cámara:**
- Detecta píxeles verdes en la imagen
- Calcula el porcentaje de verde visible
- Usado para identificar cuándo alcanza el objetivo

### 3. Sistema de Decisión

**Estrategia Epsilon-Greedy:**
- **Exploración** (ε): Acción aleatoria para descubrir nuevas estrategias
- **Explotación** (1-ε): Usa la mejor acción conocida según la tabla Q
- Epsilon comienza en 1.0 (100% exploración) y decae a 0.01 (1% exploración)

**Acciones Disponibles:**
1. **Forward**: Ambas ruedas a velocidad máxima (avanzar recto)
2. **Left**: Rueda izquierda retrocede, derecha avanza (girar izquierda)
3. **Right**: Rueda izquierda avanza, derecha retrocede (girar derecha)

---

## Sistema de Recompensas

El robot aprende mediante las siguientes recompensas:

| Evento | Recompensa | Descripción |
|--------|-----------|-------------|
| **Alcanzar objetivo** | +100 | >30% de píxeles verdes en cámara |
| **Ver objetivo parcialmente** | +0 a +5 | Proporcional a % de verde visible |
| **Acercarse a obstáculo** | -10 | Sensor detecta objeto muy cerca |
| **Cada paso** | -0.01 | Motiva a encontrar rutas eficientes |

---

## Parámetros de Aprendizaje

### Parámetros Q-Learning

```python
learning_rate = 0.1      # α - Velocidad de aprendizaje
discount_factor = 0.9    # γ - Importancia de recompensas futuras
epsilon = 1.0            # Tasa de exploración inicial
epsilon_decay = 0.995    # Reducción de epsilon por episodio
epsilon_min = 0.01       # Exploración mínima
```

### Parámetros de Episodio

```python
max_steps_per_episode = 1000    # Máximo de pasos antes de reiniciar
goal_detection_threshold = 0.3  # % de verde para considerar objetivo alcanzado
max_speed = 6.28                # Velocidad máxima de ruedas (rad/s)
```

### Ajustar Parámetros

Para modificar estos valores, edita el archivo:
`controllers/q_learning_controller/q_learning_controller.py` (líneas 20-27)

---

## Observando el Aprendizaje

### Mensajes en Consola

Durante la ejecución verás:

```
Iniciando Q-Learning...
Parámetros: α=0.1, γ=0.9, ε=1.0

Episodio 0 terminado (max pasos), Recompensa total: -10.45
Episodio 1 terminado (max pasos), Recompensa total: -8.32
¡Objetivo alcanzado! Episodio 2, Pasos: 456, Recompensa total: 89.23
Episodio 10 - Epsilon: 0.951 - Estados aprendidos: 234
```

### Fases del Aprendizaje

**Fase 1: Exploración (Episodios 1-50)**
- Comportamiento mayormente aleatorio
- Epsilon alto (>0.6)
- Muchas colisiones
- Descubrimiento del espacio de estados

**Fase 2: Aprendizaje (Episodios 50-200)**
- Balance entre exploración y explotación
- Epsilon medio (0.2-0.6)
- Comienza a evitar obstáculos mejor
- Ocasionalmente alcanza el objetivo

**Fase 3: Refinamiento (Episodios 200+)**
- Principalmente explotación
- Epsilon bajo (<0.2)
- Rutas más eficientes
- Mayor tasa de éxito

### Indicadores de Progreso

- **Estados aprendidos**: Aumenta conforme explora → Mayor conocimiento
- **Recompensa total**: Aumenta con el tiempo → Mejor desempeño
- **Pasos para alcanzar objetivo**: Disminuye → Rutas más eficientes
- **Epsilon**: Disminuye → Más confianza en lo aprendido

---

## Personalización

### Modificar el Entorno

**Agregar más obstáculos** (`worlds/proyecto-q-learning.wbt`):

```vrml
DEF WALL3 Solid {
  translation 0 0.5 0.05
  children [
    Shape {
      appearance PBRAppearance {
        baseColor 0.5 0.5 0.5
      }
      geometry Box {
        size 0.6 0.05 0.1
      }
    }
  ]
  boundingObject Box {
    size 0.6 0.05 0.1
  }
}
```

**Cambiar posición del objetivo**:
Edita la línea 32 del archivo `.wbt`:
```vrml
translation 0.7 0.7 0.05  # Cambiar coordenadas X Y Z
```

### Modificar el Comportamiento

**Agregar más acciones**:
En `q_learning_controller.py`, modifica:
```python
self.actions = ['forward', 'left', 'right', 'backward']  # Nueva acción
```

Y agrega en `execute_action()`:
```python
elif action == 'backward':
    self.left_motor.setVelocity(-self.max_speed)
    self.right_motor.setVelocity(-self.max_speed)
```

**Cambiar discretización**:
Modifica `discretize_state()` para más o menos niveles de precisión.

---

## Solución de Problemas

### El robot no se mueve
- Verifica que el controlador esté asignado al robot en el archivo `.wbt`
- Revisa que no haya errores en la consola de Webots

### No alcanza el objetivo nunca
- Aumenta `max_steps_per_episode` para darle más tiempo
- Reduce `epsilon_decay` para mantener más exploración
- Verifica que la cámara esté detectando el verde (ajusta `goal_detection_threshold`)

### Aprende muy lento
- Aumenta `learning_rate` (pero no más de 0.5)
- Ajusta el sistema de recompensas para dar más feedback positivo
- Reduce la discretización del estado (menos niveles = aprendizaje más rápido)

### Aprende pero luego empeora
- Puede ser que epsilon sea muy bajo muy rápido
- Aumenta `epsilon_min` a 0.05 o 0.1
- Reduce `epsilon_decay` a 0.998

### Error "Device not found"
- Verifica que el robot tenga `supervisor TRUE` en el `.wbt`
- Asegúrate de usar un robot E-puck estándar

---

## Próximos Pasos

Ideas para extender el proyecto:

1. **Deep Q-Learning**: Usar redes neuronales en lugar de tabla Q
2. **Múltiples objetivos**: Hacer que el robot visite varios puntos
3. **Obstáculos dinámicos**: Agregar objetos que se mueven
4. **Entorno más complejo**: Laberinto con múltiples habitaciones
5. **Visualización**: Graficar el progreso de aprendizaje en tiempo real
6. **Competencia**: Múltiples robots aprendiendo simultáneamente

---

## Autor

Jayan Caceres Cuba


