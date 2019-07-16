# Curso de introducción a Deep Learning
Curso desarrollado en el Centro de desarrollo tecnológico CreaTIC como acercamiento a las técnicas más relevantes del aprendizaje profundo

# Acerca del curso

El curso ha sido desarrollado con una metología equilibrada entre conceptos teóricos y prácticos haciendo uso de materiales como los propuestos por [Omar U. Florez](https://www.linkedin.com/in/omar-u-florez-35338015/), [François Chollet](https://www.linkedin.com/in/fchollet/) e [Ian Goodfellow](https://www.linkedin.com/in/ian-goodfellow-b7187213/).

# Instructor 
Este curso ha sido una recopilación y/o creación de [Gabriel Eduardo Rengifo](https://www.linkedin.com/in/gabriel-eduardo-rengifo-rios-685b3384/), Asesor tecnológico para el [Centro de desarrollo tecnológico CreaTIC](https://www.linkedin.com/company/cluster-creatic/) - Popayán Colombia  

## Contenido

Conceptos básicos:
* ¿Qué es Inteligencia Artificial, Machine Learning y Deep Learning?
* Ciencia de datos, diferencias y similitudes con Machine Learning
* Aproximaciones de Deep Learning
* ¿Qué es una neurona y cómo funciona?
* ¿De dónde provienen, como son y para qué sirven los datos? 
* Instalación de entorno y herramientas
  
Uso de los datos en DL:
* Repositorios de datos y preprocesamiento (Limpieza, verificación de intregridad)
* Cosas a tener en cuenta al trabajar con datos (homogeneidad, fuentes, tamaño)
* Práctica con Jupyter Notebook y Anaconda cargado grandes volúmenes de
datos.
Redes neuronales y Modelos DL:
* ¿Qué es una red neuronal artificial?
* Modelo matemático de las Redes Neuronales con TensorFlow
* Función softmax(z) y sigmoid(z)
* Lógica difusa y otras aproximaciones de IA
* Modelos con librerías de Python:
  * Logistic Classifier
  * Multi Layer Perceptron
  * Long-Short Term Memory
* Algoritmos no supervisados
* Práctica de DL – Resolución de problemas planteados.

Evaluación, integración:
* Evaluación de sistemas entrenados
* Precisión de los resultados
* Integración con otros sistemas computacionales


## Configurar el entorno de trabajo
### Pre-requesitos: 
- [Anaconda Distribution](https://www.anaconda.com/distribution/) 
- [VS Code](https://code.visualstudio.com)

Clona este repositorio y ejecuta la siguiente lista de comandos:

Usa virtual para tener un entorno de trabajo e instalar las librerías
```shell
pip install virtualenv
```
Posteriormente crea y luego activa el entorno virtual
```shell
virtualenv venv
#On Windows
./venv/Scripts/activate
#On Linux
source venv/bin/activate
```
Ahora instala los requerimientos para el proyecto
```shell
pip install -r requirements.txt
```
Finalmente instala el kernel de Jupyter dentro del virtualenv
```shell
ipython kernel install --user --name=venv
```

## Ejecutar y probar el API
Después de habilitar el entorno virtual ejecutar
```shell
cd deep_api 
python manage.py migrate
#Cargará los modelos en una pequeña base de datos local sqlite.
python manage.py runserver
#Pondrá en funcionamiento el servidor local, la aplicación estará disponible
#localhost:8000
```
Finalmente envíe una una imagen de las contenidas en el cartepa test
usando un cliente API Rest como Postman
Use el endpoint: [https://localhost:8000/api/deep/](https://localhost:8000/api/deep/) y el método POST

## Estructura del proyecto

```yaml
deeplearningcourse:
    docs:           Material en pdf
    data:           Datos de ejemplo
    models:         Modelos desarrollados
    notebook:       Archivos Jupyter 
    practices:      Prácticas por tema
    utils:          Utilidades
```

## Dataset

TO-DO

```
- MINST
- Titanic Survivors
- Spanish Emojis


```
## Licencia de uso

Apache Version 2.0, January 2004
Para más información consulte [LICENSE](LICENSE)


## Recursos y créditos
#### Libros
***
- [Deep Learning MIT Press book](https://www.deeplearningbook.org)
- [Deep Learning with Python](http://faculty.neu.edu.cn/yury/AAI/Textbook/Deep%20Learning%20with%20Python.pdf) 

#### Repositorios 
***
- [AMP-Tech](https://github.com/puigalex/AMP-Tech/) 
- [TensorFlow 101](https://github.com/serengil/tensorflow-101/tree/master/python)
- [Tensorflow ANFIS](https://github.com/tiagoCuervo/TensorANFIS)
- [Scratch_mlp](https://github.com/omar-florez/scratch_mlp)
- [Learning emotions whit emojis](https://github.com/omar-florez/learning_emotions_with_emojis)
#### Perfiles
  ***
- [@Omar-florez](https://github.com/omar-florez)
- [@andreaOtero](https://github.com/andreaOtero)
- [@tiagoCuervo](https://github.com/tiagoCuervo)
- [@serengil](https://github.com/serengil)
- [@puigalex](https://github.com/puigalex)