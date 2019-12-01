#LIBRERIAS A USAR PARA LA PREDICCIÓN
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

#VARAIBLES DE DIMENSIONES DE LAS IMAGENES A PREDECIR
longitud, altura = 224, 224 #Longitud y altura de las imagenes

#DIRECCIONES DEL MODELO Y PESOS
modelo = './modelos/modelo.h5' #Dirección del modelo
pesos_modelo = './modelo/pesos.h5' #Dirección de los pesos del modelo

#CARGAMOS EL MODELO Y LOS PESOS DE LA RED NEURONAL
cnn = load_model(modelo) #Cargamos el modelo
cnn.load_weights(pesos_modelo) #Cargamos los pesos del modelo

#FUNCIÓN PARA REALIZAR LA PREDICCIÓN DE UNA IMAGEN
def predict(file):
    x = load_img(file, target_size = (longitud, altura)) #Cargamos las imagen con longitud y altura esperados en las variables
    x = img_to_array(x) #Convertmos la imagen en un array
    x = np.expand_dims(x, axis = 0) #Expandimos sus dimensiones
    array = cnn.predict(x) #Realizamos la predicción de la imagen con lo contenido en el modelo
    result = array[0] #Tomamos el arreglo donde con las probabilidades de coincidencia de la imagen con las clases
    answer = np.argmax(result) #Toma el valor mas alto y regresa el índice en donde se encuentra
    if answer == 0: #En caso de que el valor de answer sea 0
        print('Predicción: Su piel muestra tener cáncer de tipo benigno')
    if answer == 1: #En caso de que el valor de answer sea 1
        print('Predicción: Su piel muestra tener cáncer de tipo maligno')

    return answer #Retorna el resultado de la predicción

#INGRESO DE IMAGEN Y PREDICCIÓN
image = input('Ingrese la foto de la radiografía a usar: ') #Pide la imagen a predecir
predict(image) #Realiza la función de predicción con esa imagen