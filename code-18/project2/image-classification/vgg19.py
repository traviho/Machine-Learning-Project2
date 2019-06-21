from typing import Tuple

from keras import Sequential, Model
from keras.layers import Conv2D, Activation, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.regularizers import l2


def get_model(shape: Tuple[int, int, int], n_classes: int, dropout: float = 0.5, l2_reg: float = 0.,
              layer: int = None, filters: int = None) -> Model:
    vgg19 = Sequential()

    # Layer 1 & 2
    vgg19.add(Conv2D(64, (3, 3), padding='same',
                     input_shape=shape, kernel_regularizer=l2(l2_reg)))
    vgg19.add(Activation('relu'))
    vgg19.add(ZeroPadding2D((1, 1)))
    vgg19.add(Conv2D(64, (3, 3), padding='same'))
    vgg19.add(Activation('relu'))
    vgg19.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 3 & 4
    vgg19.add(ZeroPadding2D((1, 1)))
    vgg19.add(Conv2D(128, (3, 3), padding='same'))
    vgg19.add(Activation('relu'))
    vgg19.add(ZeroPadding2D((1, 1)))
    vgg19.add(Conv2D(128, (3, 3), padding='same'))
    vgg19.add(Activation('relu'))
    vgg19.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 5, 6, 7, & 8
    vgg19.add(ZeroPadding2D((1, 1)))
    vgg19.add(Conv2D(256, (3, 3), padding='same'))
    vgg19.add(Activation('relu'))
    vgg19.add(ZeroPadding2D((1, 1)))
    vgg19.add(Conv2D(256, (3, 3), padding='same'))
    vgg19.add(Activation('relu'))
    vgg19.add(ZeroPadding2D((1, 1)))
    vgg19.add(Conv2D(256, (3, 3), padding='same'))
    vgg19.add(Activation('relu'))
    vgg19.add(ZeroPadding2D((1, 1)))
    vgg19.add(Conv2D(256, (3, 3), padding='same'))
    vgg19.add(Activation('relu'))
    vgg19.add(MaxPooling2D(pool_size=(2, 2)))

    # Layers 9, 10, 11, & 12
    vgg19.add(ZeroPadding2D((1, 1)))
    vgg19.add(Conv2D(512, (3, 3), padding='same'))
    vgg19.add(Activation('relu'))
    vgg19.add(ZeroPadding2D((1, 1)))
    vgg19.add(Conv2D(512, (3, 3), padding='same'))
    vgg19.add(Activation('relu'))
    vgg19.add(ZeroPadding2D((1, 1)))
    vgg19.add(Conv2D(512, (3, 3), padding='same'))
    vgg19.add(Activation('relu'))
    vgg19.add(ZeroPadding2D((1, 1)))
    vgg19.add(Conv2D(512, (3, 3), padding='same'))
    vgg19.add(Activation('relu'))
    vgg19.add(MaxPooling2D(pool_size=(2, 2)))

    # Layers 13, 14, 15, & 16
    vgg19.add(ZeroPadding2D((1, 1)))
    vgg19.add(Conv2D(512, (3, 3), padding='same'))
    vgg19.add(Activation('relu'))
    vgg19.add(ZeroPadding2D((1, 1)))
    vgg19.add(Conv2D(512, (3, 3), padding='same'))
    vgg19.add(Activation('relu'))
    vgg19.add(ZeroPadding2D((1, 1)))
    vgg19.add(Conv2D(512, (3, 3), padding='same'))
    vgg19.add(Activation('relu'))
    vgg19.add(ZeroPadding2D((1, 1)))
    vgg19.add(Conv2D(512, (3, 3), padding='same'))
    vgg19.add(Activation('relu'))
    vgg19.add(MaxPooling2D(pool_size=(2, 2)))

    # Layers 17, 18, & 19
    vgg19.add(Flatten())
    vgg19.add(Dense(4096))
    vgg19.add(Activation('relu'))
    vgg19.add(Dropout(0.5))
    vgg19.add(Dense(4096))
    vgg19.add(Activation('relu'))
    vgg19.add(Dropout(0.5))
    vgg19.add(Dense(n_classes))
    vgg19.add(Activation('softmax'))

    return vgg19
