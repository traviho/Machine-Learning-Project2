from typing import Tuple

from keras.datasets import mnist, cifar10
import numpy as np


def load_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[int, int, int], int]:
    number_of_classes = 10
    shape = (28, 28, 1)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((-1,) + shape).astype(np.float32)
    x_test = x_test.reshape((-1,) + shape).astype(np.float32)

    x_train /= 255.0

    x_test /= 255.0

    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)

    return x_train, y_train, x_test, y_test, shape, number_of_classes


def load_cifar() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[int, int, int], int]:
    number_of_classes = 40
    shape = (32, 32, 3)

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    x_train /= 255
    x_test /= 255

    return x_train, y_train, x_test, y_test, shape, number_of_classes
