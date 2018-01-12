import keras
from keras.models import Model, Sequential
from keras.layers import GlobalAveragePooling2D, Dense
from keras.applications.vgg16 import VGG16
from image_helpers.helpers import get_data

if __name__ == "__main__":
    X, Y, X_val, Y_val = get_data('/media/michal/USB DISK/tryp.hdf5')

    model = VGG16(include_top = False, weights='imagenet')
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(2, activation = 'softmax')(x)
    model = Model(model.input, x)
    model.summary()

    for layer in model.layers[:-1]:
        layer.trainable = False
    model.compile(optimizer="adam", loss="categorical_crossentropy",
                  metrics=["accuracy"])
    model.evaluate(X_val, Y_val)

    model.fit(X, Y, epochs=10, validation_data=[X_val, Y_val])