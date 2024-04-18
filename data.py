import numpy as np
import pathlib


def get_mnist():
    with np.load(f"{pathlib.Path(__file__).parent.absolute()}/data/mnist.npz") as f:
        images_train, labels_train, images_test = f["x_train"], f["y_train"], f["x_test"]
    images_train = images_train.astype("float32") / 255
    images_train = np.reshape(images_train, (images_train.shape[0], images_train.shape[1] * images_train.shape[2]))
    images_test = images_test.astype("float32") / 255
    images_test = np.reshape(images_test, (images_test.shape[0], images_test.shape[1] * images_test.shape[2]))
    labels_train = np.eye(10)[labels_train]
    return images_train, labels_train, images_test
