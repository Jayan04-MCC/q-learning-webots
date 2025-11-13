# Proyecto Q-Learning: Navegación Autónoma con Robot E-puck

Implementación de Q-Learning para navegación autónoma de un robot E-puck que aprende a alcanzar un objetivo verde evitando obstáculos en Webots.

**Autor**: Jayan Caceres Cuba

---

## Tabla de Contenidos

- [Evidencia de Funcionamiento](#evidencia-de-funcionamiento)
- [Arquitectura del Sistema](#arquitectura-del-sistema)
- [Componentes Esenciales del Código](#componentes-esenciales-del-código)
- [Sistema de Recompensas](#sistema-de-recompensas)
- [Ciclo de Aprendizaje](#ciclo-de-aprendizaje)
- [Parámetros Configurables](#parámetros-configurables)
- [Ejecución del Proyecto](#ejecución-del-proyecto)

---

## Evidencia de Funcionamiento

<!-- Insertar aquí captura de pantalla o GIF del robot navegando -->

![alt text](image-1.png)

**Descripción**: Robot E-puck navegando desde la posición inicial (-0.7, -0.7) hasta el objetivo verde en (0.7, 0.7) tras N episodios de entrenamiento.

---

## Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                      ENTORNO WEBOTS                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Arena 2x2m                                          │   │
│  │  ┌─────┐         WALL1         ┌─────┐              │   │
│  │  │     │           ║            │     │              │   │
│  │  │  E  │    WALL2══╣            │  🟢 │ Objetivo     │   │
│  │  │puck │           ║            │Verde│ (0.7,0.7)    │   │
│  │  └─────┘                        └─────┘              │   │
│  │  (-0.7,-0.7)                                         │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↓↑
        ┌───────────────────────────────────────┐
        │   Q-LEARNING CONTROLLER (Python)      │
        │                                       │
        │  ┌─────────────────────────────────┐ │
        │  │  1. Percepción                  │ │
        │  │     • 8 Sensores de distancia   │ │
        │  │     • Cámara 64x48 (detección)  │ │
        │  └─────────────────────────────────┘ │
        │              ↓                        │
        │  ┌─────────────────────────────────┐ │
        │  │  2. Discretización de Estado    │ │
        │  │     • 4 Direcciones × 3 Niveles │ │
        │  │     • Estado = (F,L,R,A)        │ │
        │  └─────────────────────────────────┘ │
        │              ↓                        │
        │  ┌─────────────────────────────────┐ │
        │  │  3. Decisión (Epsilon-Greedy)   │ │
        │  │     • Exploración vs Explotación│ │
        │  │     • Q-Table lookup            │ │
        │  └─────────────────────────────────┘ │
        │              ↓                        │
        │  ┌─────────────────────────────────┐ │
        │  │  4. Acción                      │ │
        │  │     • Forward / Left / Right    │ │
        │  └─────────────────────────────────┘ │
        │              ↓                        │
        │  ┌─────────────────────────────────┐ │
        │  │  5. Recompensa                  │ │
        │  │     • +100: Objetivo alcanzado  │ │
        │  │     • -10: Colisión             │ │
        │  │     • +0~1.5: Ver verde         │ │
        │  └─────────────────────────────────┘ │
        │              ↓                        │
        │  ┌─────────────────────────────────┐ │
        │  │  6. Actualización Q-Table       │ │
        │  │     • Q(s,a) ← Q(s,a) + α(...)  │ │
        │  └─────────────────────────────────┘ │
        └───────────────────────────────────────┘
```

---

## Componentes Esenciales del Código

### 1. Inicialización del Sistema
**Archivo**: `controllers/q_learning_controller/q_learning_controller.py`

```python
# Líneas 14-79: Constructor de la clase
class QLearningRobot:
    def __init__(self):
        # Parámetros de Q-Learning (líneas 21-25)
        self.learning_rate = 0.1       # α - Tasa de aprendizaje
        self.discount_factor = 0.9     # γ - Factor de descuento
        self.epsilon = 1.0             # Exploración inicial (100%)
        self.epsilon_decay = 0.995     # Decaimiento por episodio
        self.epsilon_min = 0.01        # Exploración mínima (1%)

        # Límites de episodio (líneas 62-64)
        self.max_steps_per_episode = 1000  # 16 segundos máximo

        # Tabla Q (línea 28)
        self.q_table = {}  # Diccionario: {estado: [Q(s,a1), Q(s,a2), Q(s,a3)]}
```

**Función**: Configura los hiperparámetros del algoritmo y inicializa los dispositivos del robot.

---

### 2. Percepción: Lectura de Sensores
**Líneas 99-108**

```python
def get_sensor_readings(self):
    """Obtener lecturas normalizadas de 8 sensores de distancia"""
    readings = []
    for sensor in self.distance_sensors:
        value = sensor.getValue()
        # Normalizar: 0 = lejos, 1 = muy cerca
        normalized = min(value / 4096.0, 1.0)
        readings.append(normalized)
    return readings
```

**Función**: Lee los 8 sensores infrarrojos y normaliza sus valores al rango [0, 1].

---

### 3. Discretización del Estado
**Líneas 110-130**

```python
def discretize_state(self, sensor_readings):
    """Agrupa 8 sensores en 4 direcciones con 3 niveles cada una"""
    # Agrupar sensores por dirección
    front = max(sensor_readings[0], sensor_readings[7])  # ps0, ps7
    left = max(sensor_readings[5], sensor_readings[6])   # ps5, ps6
    right = max(sensor_readings[1], sensor_readings[2])  # ps1, ps2
    back = max(sensor_readings[3], sensor_readings[4])   # ps3, ps4

    # Discretizar cada dirección en 3 niveles
    def discretize_value(value):
        if value < 0.15:
            return 0  # Libre
        elif value < 0.40:
            return 1  # Cerca
        else:
            return 2  # Muy cerca

    state = (discretize_value(front),
             discretize_value(left),
             discretize_value(right),
             discretize_value(back))

    return state  # Tupla: (F, L, R, A)
```

**Función**: Convierte las 8 lecturas continuas en un estado discreto de 4 dimensiones con 3 niveles cada una.
**Espacio de estados**: 3^4 = 81 estados posibles.

---

### 4. Detección del Objetivo (Visión)
**Líneas 170-205**

```python
def detect_goal(self):
    """Detecta píxeles verdes en la imagen de la cámara"""
    if not self.camera:
        return 0.0

    image = self.camera.getImage()
    if not image:
        return 0.0

    width = self.camera.getWidth()   # 64 píxeles
    height = self.camera.getHeight()  # 48 píxeles

    green_pixels = 0
    total_pixels = width * height

    # Analizar cada píxel
    for y in range(height):
        for x in range(width):
            # Obtener componentes RGB
            r = self.camera.imageGetRed(image, width, x, y)
            g = self.camera.imageGetGreen(image, width, x, y)
            b = self.camera.imageGetBlue(image, width, x, y)

            # Criterio de detección de verde
            if g > 150 and g > r * 1.5 and g > b * 1.5:
                green_pixels += 1

    # Retornar proporción de verde [0.0, 1.0]
    return green_pixels / total_pixels
```

**Función**: Analiza los 3,072 píxeles de la cámara y calcula el porcentaje que es verde.

---

### 5. Cálculo de Recompensas
**Líneas 207-230**

```python
def calculate_reward(self, sensor_readings):
    """Sistema de recompensas basado en sensores y visión"""

    # 1. Detectar objetivo verde
    green_ratio = self.detect_goal()

    # 2. RECOMPENSA MÁXIMA: Objetivo alcanzado
    if green_ratio > self.goal_detection_threshold:  # >30% verde
        return 100.0, True  # Episodio terminado

    # 3. PENALIZACIÓN: Colisión con obstáculo
    max_sensor = max(sensor_readings)
    if max_sensor > 0.5:  # Obstáculo muy cerca
        return -10.0, False

    # 4. RECOMPENSA PROPORCIONAL: Ver algo de verde
    # Fórmula: -0.01 (costo por paso) + 5 × (% verde)
    reward = -0.01 + (green_ratio * 5.0)

    return reward, False
```

**Tabla de Recompensas**:
| Condición | Valor | Efecto |
|-----------|-------|--------|
| `green_ratio > 0.30` | **+100.0** | Termina episodio |
| `max_sensor > 0.50` | **-10.0** | Continúa |
| `0 < green_ratio ≤ 0.30` | `-0.01 + (green_ratio × 5.0)` | Guía hacia objetivo |

---

### 6. Decisión de Acción (Epsilon-Greedy)
**Líneas 132-158**

```python
def choose_action(self, state):
    """Estrategia Epsilon-Greedy para balancear exploración/explotación"""

    # Exploración: Acción aleatoria
    if random.random() < self.epsilon:
        return random.randint(0, self.num_actions - 1)

    # Explotación: Mejor acción según Q-Table
    q_values = self.get_q_values(state)
    return np.argmax(q_values)

def get_q_values(self, state):
    """Obtiene o inicializa Q-values para un estado"""
    if state not in self.q_table:
        # Inicializar con ceros
        self.q_table[state] = np.zeros(self.num_actions)
    return self.q_table[state]
```

**Función**: Decide entre explorar (acción aleatoria) o explotar (mejor acción conocida).

---

### 7. Ejecución de Acciones
**Líneas 160-168**

```python
def execute_action(self, action_index):
    """Traduce índice de acción a comandos de motores"""
    action = self.actions[action_index]

    if action == 'forward':
        self.left_motor.setVelocity(self.max_speed)
        self.right_motor.setVelocity(self.max_speed)
    elif action == 'left':
        self.left_motor.setVelocity(-self.max_speed * 0.5)
        self.right_motor.setVelocity(self.max_speed * 0.5)
    elif action == 'right':
        self.left_motor.setVelocity(self.max_speed * 0.5)
        self.right_motor.setVelocity(-self.max_speed * 0.5)
```

**Acciones disponibles**:
- `0 = forward`: Avanzar recto
- `1 = left`: Girar a la izquierda
- `2 = right`: Girar a la derecha

---

### 8. Actualización de Q-Values
**Líneas 232-239**

```python
def update_q_value(self, state, action, reward, next_state):
    """Aplica la ecuación de Bellman para Q-Learning"""

    # Obtener Q-value actual
    current_q = self.get_q_values(state)[action]

    # Obtener mejor Q-value del siguiente estado
    next_max_q = np.max(self.get_q_values(next_state))

    # Ecuación de Q-Learning
    new_q = current_q + self.learning_rate * (
        reward + self.discount_factor * next_max_q - current_q
    )

    # Actualizar tabla Q
    self.q_table[state][action] = new_q
```

**Ecuación**:
```
Q(s,a) ← Q(s,a) + α × [r + γ × max(Q(s',a')) - Q(s,a)]
```

**Donde**:
- `Q(s,a)`: Valor actual de tomar acción `a` en estado `s`
- `α = 0.1`: Tasa de aprendizaje
- `r`: Recompensa inmediata
- `γ = 0.9`: Factor de descuento (importancia del futuro)
- `max(Q(s',a'))`: Mejor valor en el siguiente estado

---

### 9. Ciclo Principal de Aprendizaje
**Líneas 268-307**

```python
def run(self):
    """Bucle principal de entrenamiento"""

    # Obtener estado inicial
    sensor_readings = self.get_sensor_readings()
    self.current_state = self.discretize_state(sensor_readings)

    while self.robot.step(self.timestep) != -1:
        # PASO 1: Elegir acción
        self.current_action = self.choose_action(self.current_state)

        # PASO 2: Ejecutar acción
        self.execute_action(self.current_action)
        self.robot.step(self.timestep)

        # PASO 3: Observar resultado
        sensor_readings = self.get_sensor_readings()
        next_state = self.discretize_state(sensor_readings)
        reward, done = self.calculate_reward(sensor_readings)

        # PASO 4: Aprender (actualizar Q-Table)
        self.update_q_value(self.current_state, self.current_action,
                           reward, next_state)

        # PASO 5: Actualizar estado
        self.current_state = next_state
        self.total_reward += reward
        self.steps += 1

        # PASO 6: Verificar fin de episodio
        if done or self.steps >= self.max_steps_per_episode:
            print(f"Episodio {self.episode}, Pasos: {self.steps}, "
                  f"Recompensa: {self.total_reward:.2f}")
            self.reset_robot()
```

**Flujo de ejecución**:
```
Iniciar → [Observar estado] → [Elegir acción] → [Ejecutar] →
[Recibir recompensa] → [Actualizar Q-Table] → ¿Terminó?
                                                 ├─ No → Repetir
                                                 └─ Sí → Reiniciar episodio
```

---

### 10. Reinicio de Episodio
**Líneas 241-266**

```python
def reset_robot(self):
    """Prepara el siguiente episodio de entrenamiento"""

    self.episode += 1
    self.steps = 0
    self.total_reward = 0

    # Reducir epsilon (menos exploración con el tiempo)
    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # Guardar progreso cada 10 episodios
    if self.episode % 10 == 0:
        self.save_q_table()
        print(f"Epsilon: {self.epsilon:.3f} - Estados: {len(self.q_table)}")

    # Reiniciar posición física
    translation_field = self.robot_node.getField('translation')
    rotation_field = self.robot_node.getField('rotation')
    translation_field.setSFVec3f(self.initial_translation)
    rotation_field.setSFRotation(self.initial_rotation)
    self.robot_node.resetPhysics()
```

**Función**: Reinicia el robot a su posición inicial y actualiza epsilon.

---

### 11. Persistencia de Datos
**Líneas 80-97**

```python
def load_q_table(self):
    """Cargar tabla Q desde archivo al iniciar"""
    if os.path.exists(self.q_table_file):
        with open(self.q_table_file, 'rb') as f:
            self.q_table = pickle.load(f)
        print(f"Tabla Q cargada: {len(self.q_table)} estados")

def save_q_table(self):
    """Guardar tabla Q en archivo"""
    with open(self.q_table_file, 'wb') as f:
        pickle.dump(self.q_table, f)
    print(f"Tabla Q guardada: {len(self.q_table)} estados")
```

**Función**: Permite continuar el entrenamiento entre ejecuciones.

---

## Sistema de Recompensas

### Diagrama de Flujo

```
                    Ejecutar Acción
                          ↓
            ┌─────────────────────────┐
            │  Leer Sensores + Cámara │
            └─────────────────────────┘
                          ↓
            ┌─────────────────────────┐
            │  ¿green_ratio > 0.30?   │
            └─────────────────────────┘
                    ↓          ↓
                  SÍ           NO
                   ↓            ↓
         ┌─────────────┐  ┌────────────────┐
         │ Recompensa: │  │ ¿max_sensor    │
         │   +100.0    │  │   > 0.50?      │
         │ Episodio    │  └────────────────┘
         │ TERMINA     │       ↓        ↓
         └─────────────┘      SÍ        NO
                              ↓          ↓
                    ┌──────────────┐  ┌───────────────────┐
                    │ Recompensa:  │  │ Recompensa:       │
                    │   -10.0      │  │ -0.01 + (green×5) │
                    └──────────────┘  └───────────────────┘
```

### Tabla Detallada

| Evento | Condición | Recompensa | Línea | Termina Episodio |
|--------|-----------|-----------|-------|------------------|
| **Objetivo alcanzado** | `green_ratio > 0.30` | `+100.0` | 219-220 | ✅ Sí |
| **Colisión inminente** | `max_sensor > 0.50` | `-10.0` | 223-225 | ❌ No |
| **Ver objetivo (30%)** | `green_ratio = 0.30` | `+1.49` | 228 | ❌ No |
| **Ver objetivo (20%)** | `green_ratio = 0.20` | `+0.99` | 228 | ❌ No |
| **Ver objetivo (10%)** | `green_ratio = 0.10` | `+0.49` | 228 | ❌ No |
| **Sin objetivo visible** | `green_ratio = 0` | `-0.01` | 228 | ❌ No |

---

## Ciclo de Aprendizaje

### Progresión de Epsilon

```
Episodio 0:    ε = 1.000  (100% exploración)
Episodio 10:   ε = 0.951  (95% exploración)
Episodio 50:   ε = 0.778  (78% exploración)
Episodio 100:  ε = 0.606  (61% exploración)
Episodio 200:  ε = 0.367  (37% exploración)
Episodio 500:  ε = 0.081  (8% exploración)
Episodio 1000: ε = 0.010  (1% exploración) ← Mínimo
```

### Evolución Típica del Aprendizaje

**Fase 1: Exploración Caótica (Episodios 1-100)**
- Comportamiento aleatorio predominante
- Muchas colisiones con obstáculos
- Recompensas totales negativas
- Tabla Q crece rápidamente (descubrimiento)

**Fase 2: Aprendizaje Activo (Episodios 100-300)**
- Balance exploración/explotación
- Comienza a evitar obstáculos
- Ocasionalmente alcanza el objetivo
- Recompensas totales aumentan gradualmente

**Fase 3: Refinamiento (Episodios 300+)**
- Comportamiento mayormente explotativo
- Rutas eficientes y consistentes
- Alta tasa de éxito en alcanzar objetivo
- Número de pasos disminuye

---

## Parámetros Configurables

### Archivo: `q_learning_controller.py`

| Parámetro | Línea | Valor Default | Descripción | Efecto al Aumentar |
|-----------|-------|---------------|-------------|-------------------|
| `learning_rate` | 21 | `0.1` | Velocidad de actualización de Q-values | Aprende más rápido pero menos estable |
| `discount_factor` | 22 | `0.9` | Importancia de recompensas futuras | Mayor planificación a largo plazo |
| `epsilon` | 23 | `1.0` | Exploración inicial | Mayor aleatoriedad al inicio |
| `epsilon_decay` | 24 | `0.995` | Velocidad de reducción de ε | Reduce exploración más lentamente |
| `epsilon_min` | 25 | `0.01` | Exploración mínima | Mantiene más exploración siempre |
| `max_steps_per_episode` | 64 | `1000` | Pasos antes de timeout (16s) | Da más tiempo para encontrar objetivo |
| `goal_detection_threshold` | 59 | `0.3` | % verde para declarar éxito | Requiere estar más cerca del objetivo |
| `max_speed` | 43 | `6.28` | Velocidad máxima (rad/s) | Robot más rápido |

### Archivo: `proyecto-q-learning.wbt`

| Parámetro | Línea | Valor | Descripción |
|-----------|-------|-------|-------------|
| `basicTimeStep` | 8 | `16` | Milisegundos por step de simulación |
| Robot translation | 12 | `[-0.7, -0.7, 0]` | Posición inicial |
| Objetivo translation | 33 | `[0.7, 0.7, 0.05]` | Posición del objetivo |
| Camera fieldOfView | 26 | `1.0` | Ángulo de visión (radianes) |
| Camera width | 27 | `64` | Ancho de imagen (píxeles) |
| Camera height | 28 | `48` | Alto de imagen (píxeles) |

---

## Ejecución del Proyecto

### Requisitos

- **Webots R2025a** o superior
- **Python 3.8+** (incluido con Webots)
- Librerías: `numpy`, `pickle` (incluidas en Webots)

### Comandos

```bash
# 1. Abrir Webots
# 2. Cargar mundo
File > Open World > D:\proyectos_webots\q-learning\worlds\proyecto-q-learning.wbt

# 3. Iniciar simulación
Play ▶️

# 4. Observar consola
Iniciando Q-Learning...
Parámetros: α=0.1, γ=0.9, ε=1.0
Episodio 0 terminado (max pasos), Recompensa total: -10.45
...

# 5. Para reiniciar desde cero
# Eliminar archivo: q_table.pkl
```

### Salida Esperada

```
Iniciando Q-Learning...
Parámetros: α=0.1, γ=0.9, ε=1.0
Episodio 0 terminado (max pasos), Recompensa total: -12.34
Episodio 1 terminado (max pasos), Recompensa total: -8.76
Episodio 2 terminado (max pasos), Recompensa total: -15.23
...
Episodio 10 - Epsilon: 0.951 - Estados aprendidos: 45
...
¡Objetivo alcanzado! Episodio 23, Pasos: 687, Recompensa total: 78.45
...
Episodio 100 - Epsilon: 0.606 - Estados aprendidos: 81
¡Objetivo alcanzado! Episodio 104, Pasos: 342, Recompensa total: 93.12
```

---

