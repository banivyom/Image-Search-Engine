from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Sequential
from keras.optimizers import SGD,Adam
from keras.models import load_model
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Flatten,Dense,BatchNormalization,Activation,Dropout

def model():

    model = load_model('mymodel.hdf5')

    reduced_model=Sequential()
    for layer in model.layers[:-5]:
        reduced_model.add(layer)
    reduced_model.add(Flatten())

    return reduced_model