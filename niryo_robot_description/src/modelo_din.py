import rbdl
import numpy as np
from functions import *

# Lectura del modelo del robot a partir de URDF (parsing)
modelo = rbdl.loadModel("/home/user/proyecto_ws/src/proyecto_niryo_ned/niryo_robot_description/urdf/ned2/niryo_ned2.urdf")

# Grados de libertad
ndof = modelo.q_size 

# Configuracion articular
q = np.array([0.75, 0.3, 0.53, 0.8, 0.25, 0.1])
# Velocidad articular
dq = np.array([0.98, 0.75, 0.8, 0.63, 0.49, 1.6])
# Aceleracion articular
ddq = np.array([0.2, 0.5, 0.4, 0.3, 1.0, 0.5])

# Arrays numpy
zeros = np.zeros(ndof)          # Vector de ceros
tau   = np.zeros(ndof)          # Para torque
g     = np.zeros(ndof)          # Para la gravedad
c     = np.zeros(ndof)          # Para el vector de Coriolis+centrifuga
M     = np.zeros([ndof, ndof])  # Para la matriz de inercia
e     = np.eye(6)               # Vector identidad
mi    = np.zeros(ndof)

# Torque dada la configuracion del robot
rbdl.InverseDynamics(modelo, q, dq, ddq, tau)

#===========================================================================================

# Parte 1: Calcular vector de gravedad, vector de Coriolis/centrifuga,
# y matriz M usando solamente InverseDynamics

    # Vector de gravedad
rbdl.InverseDynamics(modelo, q, zeros, zeros, g)
g = np.round(g,4)
print('VECTOR GRAVEDAD')
print(g)

    # Vector de Coriolis/centrifuga
rbdl.InverseDynamics(modelo, q, dq, zeros, c)
c = c-g
c= np.round(c,4)
print('VECTOR F y C')
print(c)

    # Matriz M
for i in range(ndof):
    rbdl.InverseDynamics(modelo, q, zeros, e[i,:], mi)
    M[i,:] = mi - g
print('MATRIZ INERCIA')
print(np.round(M,4))
print('==============================')
#===========================================================================================

# Parte 2: Calcular M y los efectos no lineales b usando las funciones
# CompositeRigidBodyAlgorithm y NonlinearEffects. Almacenar los resultados
# en los arreglos llamados M2 y b2

b2 = np.zeros(ndof)          # Para efectos no lineales
M2 = np.zeros([ndof, ndof])  # Para matriz de inercia

rbdl.CompositeRigidBodyAlgorithm(modelo, q, M2)
rbdl.NonlinearEffects(modelo, q, dq, b2)
print('VECTOR DE EFECTOS NO LINEALES')
print(np.round(b2,4))

print('MATRIZ DE INERCIA')
print(np.round(M2,4))

print('==============================')
#===========================================================================================
# Parte 2: Verificacion de valores

    # Resta de vectores para observar la comprobacion
error1 = b2 - c - g
print('1ra VERIFICACION')
print(np.round(error1, 4))

    # Resta de matrices de inercias obtenidas en ambos puntos
error1 = M2 - M
print('2da VERIFICACION')
print(np.round(error1, 4))

print('==============================')
#===========================================================================================
# Parte 3: Verificacion de la expresion de la dinamica

tau2 = M.dot(ddq) + c + g
tau3 = M2.dot(ddq) + b2

print('Vector de torques obtenidos con la funcion InverseDynamics')
print(np.round(tau,4))
print('Vector de torques obtenidos en la primera parte con la funcion InverseDynamics')
print(np.round(tau2,4))
print('Vector de torques obtenidos en la segunda parte con las funciones CompositeRigidAlgorithm y NonlinearEffects')
print(np.round(tau3,4))
