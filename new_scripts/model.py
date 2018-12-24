import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Conv2D
from keras.applications.vgg16 import VGG16


def make_cut_vgg(layer_name, input_s, double_dense = False):
    model = VGG16(include_top = False, input_shape=(input_s, input_s, 3), weights='imagenet')
    x = model.get_layer(layer_name).output
    layer_size = int(x.get_shape()[-1])
    x = Conv2D(layer_size, 1)(x)
    x = Flatten()(x)
    x = Dense(layer_size, activation='relu', name='fc1')(x)
    if double_dense == True:
        x = Dropout(0.5)(x)
        x = Dense(layer_size, activation='relu', name='fc2')(x)
    x = Dense(1, activation = 'sigmoid')(x)
    model = Model(model.input, x)
    #for layer in model.layers[:-7]:
    #    layer.trainable = False

    return model

def make_vgg():
    model = VGG16(include_top = False, weights=None)
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1, activation = 'sigmoid')(x)
    model = Model(model.input, x)
    #for layer in model.layers[:-1]:
    #    layer.trainable = False
    
def make_vgg(double_dense=True):
    model = VGG16(include_top = False, input_shape=(224, 224, 3), weights='imagenet')
    x = model.output
    x = Flatten(name='flatten')(x)
    if double_dense == True:
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dropout(0.5)(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(1, activation = 'sigmoid')(x)
    model = Model(model.input, x)
#    for layer in model.layers[:-5]:
#        layer.trainable = False

    return model