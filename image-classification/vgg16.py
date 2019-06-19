from typing import Tuple

from keras import Sequential, Model
from keras.activations import relu, softmax
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.regularizers import l2


def get_model(shape: Tuple[int, int, int], n_classes: int, dropout: float = 0.5, l2_reg: float = 0.,
              layer: int = None, filters: int = None) -> Model:
    model = Sequential()

    # Layer 1 & 2
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=shape, kernel_regularizer=l2(l2_reg)))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # Layer 3 & 4
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 5, 6, & 7
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Layers 8, 9, & 10
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Layers 11, 12, & 13
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Layers 14, 15, & 16
    model.add(Flatten())
    model.add(Dense(4096, activation=relu))
    # vgg16.add(Dropout(0.5))
    # vgg16.add(Dense(4096, activation=relu))
    # vgg16.add(Dropout(0.5))
    model.add(Dense(n_classes, activation=softmax))

    return model
