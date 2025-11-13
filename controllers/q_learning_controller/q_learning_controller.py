"""
Q-Learning Controller para Robot E-puck en Webots
Este controlador implementa el algoritmo Q-Learning para que un robot
aprenda a navegar desde una posición inicial hasta un objetivo (caja verde)
evitando obstáculos.
"""

from controller import Supervisor
import numpy as np
import random
import pickle
import os

class QLearningRobot:
    def __init__(self):
        # Inicializar el robot como Supervisor para poder reiniciar la simulación
        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())

        # Parámetros de Q-Learning
        self.learning_rate = 0.1  # Alpha
        self.discount_factor = 0.9  # Gamma
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        # Tabla Q: diccionario donde la clave es el estado y el valor es un array de Q-values por acción
        self.q_table = {}

        # Definir acciones posibles
        self.actions = ['forward', 'left', 'right']
        self.num_actions = len(self.actions)

        # Inicializar motores
        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # Velocidades
        self.max_speed = 6.28

        # Inicializar sensores de distancia (E-puck tiene 8 sensores)
        self.distance_sensors = []
        sensor_names = ['ps0', 'ps1', 'ps2', 'ps3', 'ps4', 'ps5', 'ps6', 'ps7']
        for name in sensor_names:
            sensor = self.robot.getDevice(name)
            sensor.enable(self.timestep)
            self.distance_sensors.append(sensor)

        # Inicializar cámara para detectar objetivo verde
        self.camera = self.robot.getDevice('camera')
        if self.camera:
            self.camera.enable(self.timestep)

        # Umbral para detectar objetivo (porcentaje de píxeles verdes en la imagen)
        self.goal_detection_threshold = 0.3

        # Variables de episodio
        self.episode = 0
        self.steps = 0
        self.max_steps_per_episode = 1000
        self.total_reward = 0

        # Variables de estado
        self.current_state = None
        self.current_action = None

        # Archivo para guardar la tabla Q
        self.q_table_file = 'q_table.pkl'
        self.load_q_table()

        # Obtener nodo del robot para reiniciar posición
        self.robot_node = self.robot.getSelf()
        self.initial_translation = [-0.7, -0.7, 0]
        self.initial_rotation = [0, 0, 1, 0]

    def load_q_table(self):
        """Cargar tabla Q desde archivo si existe"""
        if os.path.exists(self.q_table_file):
            try:
                with open(self.q_table_file, 'rb') as f:
                    self.q_table = pickle.load(f)
                print(f"Tabla Q cargada con {len(self.q_table)} estados")
            except:
                print("No se pudo cargar la tabla Q, iniciando nueva")

    def save_q_table(self):
        """Guardar tabla Q en archivo"""
        try:
            with open(self.q_table_file, 'wb') as f:
                pickle.dump(self.q_table, f)
            print(f"Tabla Q guardada con {len(self.q_table)} estados")
        except:
            print("No se pudo guardar la tabla Q")

    def get_sensor_readings(self):
        """Obtener lecturas de los sensores de distancia"""
        readings = []
        for sensor in self.distance_sensors:
            value = sensor.getValue()
            # Normalizar el valor (los sensores del E-puck devuelven valores entre 0 y ~4000)
            normalized = min(value / 4000.0, 1.0)
            readings.append(normalized)
        return readings

    def discretize_state(self, sensor_readings):
        """
        Discretizar el estado basado en las lecturas de sensores
        Dividimos los sensores en grupos: frente, izquierda, derecha, atrás
        y categorizamos cada grupo en: libre, cerca, muy_cerca
        """
        # Agrupar sensores
        front = max(sensor_readings[0], sensor_readings[7])  # ps0, ps7
        left = max(sensor_readings[5], sensor_readings[6])   # ps5, ps6
        right = max(sensor_readings[1], sensor_readings[2])  # ps1, ps2
        back = max(sensor_readings[3], sensor_readings[4])   # ps3, ps4

        # Discretizar cada dirección
        def discretize_reading(value):
            if value < 0.15:
                return 0  # libre
            elif value < 0.4:
                return 1  # cerca
            else:
                return 2  # muy_cerca

        state = (
            discretize_reading(front),
            discretize_reading(left),
            discretize_reading(right),
            discretize_reading(back)
        )

        return state

    def get_q_values(self, state):
        """Obtener Q-values para un estado dado"""
        if state not in self.q_table:
            # Inicializar con valores aleatorios pequeños
            self.q_table[state] = np.random.uniform(-0.1, 0.1, self.num_actions)
        return self.q_table[state]

    def choose_action(self, state):
        """Seleccionar acción usando estrategia epsilon-greedy"""
        if random.random() < self.epsilon:
            # Exploración: acción aleatoria
            return random.randint(0, self.num_actions - 1)
        else:
            # Explotación: mejor acción según Q-table
            q_values = self.get_q_values(state)
            return np.argmax(q_values)

    def execute_action(self, action_index):
        """Ejecutar la acción seleccionada"""
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

    def detect_goal(self):
        """
        Detectar si el robot ve el objetivo verde usando la cámara
        Retorna el porcentaje de píxeles verdes en la imagen
        """
        if not self.camera:
            if self.steps == 1:  # Solo mostrar una vez por episodio
                print("  [WARNING] Cámara no disponible")
            return 0.0

        # Obtener imagen de la cámara
        image = self.camera.getImage()
        if not image:
            if self.steps == 1:
                print("  [WARNING] No se puede obtener imagen de la cámara")
            return 0.0

        width = self.camera.getWidth()
        height = self.camera.getHeight()

        # Contar píxeles verdes
        green_pixels = 0
        total_pixels = width * height

        for i in range(width):
            for j in range(height):
                # Obtener color del píxel
                r = self.camera.imageGetRed(image, width, i, j)
                g = self.camera.imageGetGreen(image, width, i, j)
                b = self.camera.imageGetBlue(image, width, i, j)

                # Detectar verde (g > r y g > b)
                if g > 150 and g > r * 1.5 and g > b * 1.5:
                    green_pixels += 1

        return green_pixels / total_pixels if total_pixels > 0 else 0.0

    def calculate_reward(self, sensor_readings):
        """
        Calcular recompensa basada en el estado actual
        """
        # Detectar objetivo verde con la cámara
        green_ratio = self.detect_goal()

        # Debug: Mostrar cuando ve verde
        if green_ratio > 0.05:  # Si ve más del 5% de verde
            print(f"  [VERDE DETECTADO] {green_ratio*100:.1f}% de la imagen - Recompensa: {-0.01 + (green_ratio * 5.0):.2f}")

        # Recompensa por llegar al objetivo (ver mucho verde)
        if green_ratio > self.goal_detection_threshold:
            return 100.0, True  # Recompensa grande y episodio terminado

        # Penalización por colisión (sensor muy cerca de obstáculo)
        max_sensor = max(sensor_readings)
        if max_sensor > 0.5:
            return -10.0, False  # Penalización por acercarse mucho a obstáculo

        # Pequeña recompensa por ver algo de verde (guía hacia el objetivo)
        reward = -0.01 + (green_ratio * 5.0)

        return reward, False

    def update_q_value(self, state, action, reward, next_state):
        """Actualizar Q-value usando la ecuación de Q-Learning"""
        current_q = self.get_q_values(state)[action]
        next_max_q = np.max(self.get_q_values(next_state))

        # Q(s,a) = Q(s,a) + α * (r + γ * max(Q(s',a')) - Q(s,a))
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_max_q - current_q)
        self.q_table[state][action] = new_q

    def reset_robot(self):
        """Reiniciar el robot para un nuevo episodio"""
        self.episode += 1
        self.steps = 0
        self.total_reward = 0

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Guardar tabla Q cada 10 episodios
        if self.episode % 10 == 0:
            self.save_q_table()
            print(f"Episodio {self.episode} - Epsilon: {self.epsilon:.3f} - Estados aprendidos: {len(self.q_table)}")

        # Reiniciar posición y rotación del robot
        translation_field = self.robot_node.getField('translation')
        rotation_field = self.robot_node.getField('rotation')
        translation_field.setSFVec3f(self.initial_translation)
        rotation_field.setSFRotation(self.initial_rotation)

        # Detener los motores
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # Reiniciar física del robot
        self.robot_node.resetPhysics()

    def run(self):
        """Ciclo principal de Q-Learning"""
        print("Iniciando Q-Learning...")
        print(f"Parámetros: α={self.learning_rate}, γ={self.discount_factor}, ε={self.epsilon}")

        # Esperar a que los sensores estén listos
        self.robot.step(self.timestep)

        # Obtener estado inicial
        sensor_readings = self.get_sensor_readings()
        self.current_state = self.discretize_state(sensor_readings)

        while self.robot.step(self.timestep) != -1:
            # Seleccionar y ejecutar acción
            self.current_action = self.choose_action(self.current_state)
            self.execute_action(self.current_action)

            # Esperar a que se ejecute la acción
            self.robot.step(self.timestep)

            # Obtener nuevo estado y recompensa
            sensor_readings = self.get_sensor_readings()
            next_state = self.discretize_state(sensor_readings)
            reward, done = self.calculate_reward(sensor_readings)

            # Actualizar Q-table
            self.update_q_value(self.current_state, self.current_action, reward, next_state)

            # Actualizar estado
            self.current_state = next_state
            self.total_reward += reward
            self.steps += 1

            # Verificar condiciones de finalización del episodio
            if done:
                print(f"¡Objetivo alcanzado! Episodio {self.episode}, Pasos: {self.steps}, Recompensa total: {self.total_reward:.2f}")
                self.reset_robot()
            elif self.steps >= self.max_steps_per_episode:
                print(f"Episodio {self.episode} terminado (max pasos), Recompensa total: {self.total_reward:.2f}")
                self.reset_robot()

# Crear y ejecutar el robot
if __name__ == "__main__":
    robot = QLearningRobot()
    robot.run()
