from keras.models import load_model
import tensorflow as tf
import os
#AQUI ME EQUIVOQUE POR ESO NO FUNCIONABA EL MODELO 
dirname = os.path.dirname(__file__)
#Configure aqui las rutas para los distintos modelos a usar
model = os.path.join(dirname, '../../save/models/MLP')
weights = os.path.join(dirname, '../../save/weights/weights.mlp.hdf5')

def load_model_from_path(path_model,path_weigth):
    loaded_model = load_model(path_model)
    loaded_model.load_weights(path_weigth)
    return loaded_model 

def load_all_models():
    graph = tf.get_default_graph()
    nn_models_dict = dict()
    nn_models_dict = {
        'MLP' : load_model_from_path(model, weights)
        #Agregar al diccionarios los modelos guardados
    }
    return nn_models_dict, graph