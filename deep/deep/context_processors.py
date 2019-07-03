from keras.models import load_model
import tensorflow as tf
import os
dirname = os.path.dirname(__file__)
model = os.path.join(dirname, '../../save/models')
weights = os.path.join(dirname, '../../save/weights/weights.epoch.hdf5')

def load_model_from_path(path_model,path_weigth):
    loaded_model = load_model(path_model)
    loaded_model.load_weights(path_weigth)
    return loaded_model 

def load_all_models():
    graph = tf.get_default_graph()
    nn_models_dict = dict()
    nn_models_dict = {
        'MLP' : load_model_from_path(model, weights)
        #Poner aqui los otros modelos a agregar al API
    }
    return nn_models_dict, graph