from typing import Tuple

from keras.models import Sequential, Model
from keras.activations import relu, softmax
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import BatchNormalization
from keras.regularizers import l2


def get_model(shape: Tuple[int, int, int], n_classes: int, dropout: float = 0.5, l2_reg: float = 0.,
              layer: int = None, filters: int = None) -> Model:
    model = Sequential()

    # Layer 1
    model.add(Conv2D(96, (11, 11), input_shape=shape,
                     padding='same', kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(Activation(relu))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 2
    model.add(Conv2D(256, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(relu))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 3
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(relu))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 4
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(1024, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(relu))

    # Layer 5
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(1024, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(relu))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 6
    model.add(Flatten())
    model.add(Dense(3072))
    model.add(BatchNormalization())
    model.add(Activation(relu))
    model.add(Dropout(dropout))

    # Layer 7
    model.add(Dense(4096))
    model.add(BatchNormalization())
    model.add(Activation(relu))
    model.add(Dropout(dropout))

    # Layer 8
    model.add(Dense(n_classes))
    model.add(BatchNormalization())
    model.add(Activation(softmax))

    return model
