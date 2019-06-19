import keras
import tensorflow as tf

from train import train_model

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth = True
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))


def main() -> None:
    batch_size = 128
    epochs = 30
    learning_rate = 0.00001
    filters = 512
    dropout = 0.5
    layers = 3
    l2_reg = 0.
    # model_name = 'rmdl_mod'  # model_name = sys.argv[2]
    dataset_name = 'cifar'  # dataset_name = sys.argv[1]

    for model_name in ['rmdl_mod']:
        try:
            train_model(
                dataset_name=dataset_name,
                model_name=model_name,
                batch_size=batch_size,
                epochs=epochs,
                learning_rate=learning_rate,
                dropout=dropout,
                l2_reg=l2_reg,
                layers=layers,
                filters=filters,
            )
        except Exception as e:
            print(e)

    # rmdl_mod
    # 1: dropout = 0.05
    # 2: dropout = 0.8
    # 3: dropout = 0.5
    # 4: dropout = 0.4
    # 5: lr = 0.0001
    # 6: 0.00001


if __name__ == "__main__":
    main()
