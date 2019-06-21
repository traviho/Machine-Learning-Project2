import os

import keras

from load import load_mnist, load_cifar

import rmdl_mod, alexnet, lenet, vgg16, vgg19, vgg16_2, vgg16_3, vgg19_3

dataset_load_dict = {
    'mnist': load_mnist,
    'cifar': load_cifar
}

model_dict = {
    'rmdl_mod': rmdl_mod,
    'lenet': lenet,
    'alexnet': alexnet,
    'vgg16': vgg16,
    'vgg19': vgg19,
    'vgg16_2': vgg16_2,
    'vgg16_3': vgg16_3,
    'vgg19_3': vgg19_3

}


def train_model(
        dataset_name: str,
        model_name: str,
        batch_size: int,
        epochs: int,
        learning_rate: float,
        dropout: float,
        l2_reg: float,
        layers: int,
        filters: int
) -> None:
    weight_dir = 'weights'
    history_dir = 'histories'
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    if not os.path.exists(history_dir):
        os.makedirs(history_dir)
    weight_file_path = os.path.join(weight_dir, dataset_name + '-' + model_name + '.hdf5')
    history_file_path = os.path.join(history_dir, dataset_name + '-' + model_name + '.txt')

    x_train, y_train, x_test, y_test, shape, number_of_classes = dataset_load_dict[dataset_name]()

    y_train = keras.utils.to_categorical(y_train, number_of_classes)
    y_test = keras.utils.to_categorical(y_test, number_of_classes)

    model = model_dict[model_name].get_model(
        shape=shape, n_classes=number_of_classes, dropout=dropout, l2_reg=l2_reg, layer=layers, filters=filters)

    model.summary()

    print('Model: ', model_name, 'Dataset: ', dataset_name)
    if os.path.exists(weight_file_path):
        print('Loading weight...')
        model.load_weights(weight_file_path)
    else:
        print('Weight does not exist')

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.adam(lr=learning_rate),
                  metrics=['accuracy'])



    callbacks = [keras.callbacks.ModelCheckpoint(
        weight_file_path, monitor='val_acc',
        verbose=1,
        save_best_only=True,
        save_weights_only=True
    )]

    history = model.fit(x_train, y_train,
                        validation_data=(x_test, y_test),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=callbacks,
                        verbose=1)

    with open(history_file_path, 'a') as f:
        f.write(str(history.history) + '\n')
