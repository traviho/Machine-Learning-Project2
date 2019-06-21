import itertools
from typing import Tuple

import numpy as np
from keras.activations import relu, softmax
from keras.callbacks import History
from keras.constraints import maxnorm
from keras.layers import Dense, Flatten
from keras.layers import MaxPooling2D, Dropout, Conv2D, Activation
from keras.models import Sequential, Model
from matplotlib import pyplot as plt


def get_model(shape: Tuple[int, int, int], n_classes: int, dropout: float = 0.5, l2_reg: float = None,
              layer: int = 3, filters: int = 512, ) -> Model:
    model = Sequential()
    model.add(Conv2D(filters, (3, 3), padding='same', input_shape=shape))
    model.add(Activation(relu))
    model.add(Conv2D(filters, (3, 3)))
    model.add(Activation(relu))

    for i in range(layer):
        model.add(Conv2D(filters, (3, 3), padding='same', activation=relu))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout))

    model.add(Flatten())
    model.add(Dense(256, activation=relu))
    model.add(Dropout(dropout))
    model.add(Dense(n_classes, activation=softmax, kernel_constraint=maxnorm(3)))

    return model


def RMDL_epoch(history: History) -> None:
    caption = ['RDL']
    plt.legend(caption, loc='upper right')
    plt.plot(history.history['acc'])
    plt.title('model train accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(caption, loc='upper right')
    plt.show()
    plt.plot(history.history['val_acc'])
    plt.title('model test accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()
    plt.legend(caption, loc='upper right')
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model train loss ')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(caption, loc='upper right')
    plt.show()
    plt.legend(caption, loc='upper right')
    # summarize history for loss
    plt.plot(history.history['val_loss'])
    plt.title('model loss test')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(caption, loc='upper right')
    plt.show()
    plt.savefig("examples.png")


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.get_cmap('Blues')):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    # else:
    # print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
