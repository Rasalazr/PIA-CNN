#LIBRERIAS A USAR PARA EL EMTRENAMIENTO
import sys
import os
import time
import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import applications

#FUNCIÓN QUE GUARDA EL MODELO DE LA RED NEURONAL
def modelo():
    vgg = applications.vgg16.VGG16() #Cargamos un modelo con capas ocultas para predecir imagenes
    cnn = Sequential() #Creamos una pila lineal de capas de tipo sequencial
    for capa in vgg.layers:
        cnn.add(capa) #Agregamos todos las capas ocultas de vgg a cnn
    cnn.layers.pop() #Quitamos el último elemento de cnn (con 1000 clases precargadas)
    for layer in cnn.layers:
        layer.trainable = False #Hacemos que el contenido de cnn no sea entrenable
    cnn.add(Dense(3, activation = 'softmax')) #Agregamos la capa que vamos a usar para la predicción con el número de clases que hay
    
    return cnn #Regresamos el modelos con las capas preecargadas y la capa de predicción que creamos

K.clear_session() #Función de envoltura para limpiar después de las pruebas de TensorFlow

#INGRESA LOS DATOS DE VALIDACIÓN Y ENTRENAMIENTO
data_train = './data/train' #Carpeta de datos de entrenamiento
data_test = './data/test' #Carpeta de datos de validación

print("\nLectura de las directorios completa\n")

#VARIABLES DE ENTRENAMIENTO Y VALIDACIÓN
epocas = 20 #Veces que va a entrenar el modelo
longitud, altura = 224, 224 #Longitud que las imagenes van a tomar para el entrenamiento
batch_size = 32 #Tamaño del lote fijo para las entradas
pasos = 1000 #Número de pasos que va a realizar el modelo
test_steps = 300 #Número de pasos que va a realizar en la validación
lr = 0.0004 #Tasa de aprendizaje

#PREPARAMOS LAS IMAGENES
#Reescalamiento para las imagenes de entrenamiento a una escala de 255 y configura para que pueda leer tanto imagenes en zoom como volteadas
train_datagen = ImageDataGenerator(
    rescale = 1. / 255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True) 

#Reescala para las imagenes de prueba a una escala de 255
test_datagen = ImageDataGenerator(rescale = 1. / 255) 

#Generamos las imagenes de entrenamiento
train_generador = train_datagen.flow_from_directory(
    data_train, 
    target_size = (altura, longitud), 
    batch_size = batch_size, 
    class_mode = 'categorical') 

#Generamos las imagenes de validación
test_generador = test_datagen.flow_from_directory(
    data_test, 
    target_size = (altura, longitud), 
    batch_size = batch_size, 
    class_mode = 'categorical') 

print("\nLectura de imagenes completa\n")

#CREA LA RED NEURONAL VGG16
tic = time.clock() #Tiempo de inicio del entrenamiento

cnn = modelo() #Cargamos el modelo de red neuronal convolucional (VGG16)

#Configuramos el proceso de aprendizaje
cnn.compile(
    loss = 'categorical_crossentropy', #Función de pérdida
    optimizers = optimizers.Adam(lr = lr), #Cadena del optimizador
    metrics = ['accuracy']) #Lista de métricas por presición

#ENTRENAMOS EL MODELO
history = cnn.fit_generator(
    train_generador, #Data a entrenar
    steps_per_epoch = pasos, #Pasos a dar en la eopca
    epochs = epocas, #Epocas del entrenamiento
    validation_data = test_generador, #Data de validación
    validation_steps = test_steps) #Pasos de validación

toc = time.clock() #Tiempo de finalización del entrenamiento
print('\n')
print(history.history.keys())

#TIEMPO DE ENTRENAMIENTO
time = toc - tic #Tiempo total del entrenamiento en segundos
days = abs(time/86400) #Obiene los dias que tardo el entrenamiento
ts1 = time - (days*86400) #Resta el tiempo en segundos y los dias convertidos en segundos
hours = abs(ts1/3600) #Obtiene las horas que duro el entrenamiento
ts2 = ts1 - (hours*3600) #Resta el restante en segundos y las horas convertidas en segundos
minutes = abs(ts2/60) #Obtiene los minutos que duro el entrenamiento de acuerdo al sobrante de las horas
ts3 = ts2 - (minutes*60) #Resta el sobrante en segundos y los minutos convertidas en segundos
seconds = abs(ts3/60) #Obtiene los segundos del sobrante
print("\nEntrenamiento completo")
print("Tiempo de entrenamiento: "+str(days)+" días, "+str(hours)+" hrs, "+str(minutes)+" mins, "+str(seconds)+" sec\n")

acc = history.evaluate(train_generator, test_generator, verbose = 1) #Porcentaje de presición
err = 1 - acc #Porcentaje de presición
print("\nPorcentaje de presición: "+str(acc*100)+" %")
print("\nPorcentaje de error: "+str(err)+" %")
      
#GENERAMOS UN ARCHIVO QUE GUARDA EL MODELO Y LOS PESOS DEL MISMO
target_dir = '.modelos/' #Dirección de la carpeta que guardará el modelo
if not os.path.exists(target_dir): #Si la carpeta en la variable target_dir existe
    os.mkdir(target_dir) #Genera la carpeta que guarda el modelo con la dirección de la variable target_dir
cnn.save('.modelos/modelo.h5') #Genera un archivo con el modelo entrenado
cnn.save_weights('.modelos/pesos.h5') #Genera un archivo con los pesos del modelo

print("Archivos del modelo generados\n")

#GRÁFICA DE PRESICIÓN EN EL ENTRENAMENIENTO / PRUEBA
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Precisión del modelo')
plt.ylabel('Precision')
plt.xlabel('Epocas')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#GRÁFICA DE PRESICIÓN Y PÉRDIDA EN EL ENTRENAMENIENTO / PRUEBA
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Pérdida del modelo')
plt.ylabel('Pérdida')
plt.xlabel('Epocas')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()