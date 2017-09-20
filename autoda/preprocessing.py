import numpy as np
import logging
import keras
from keras import backend as K

__all__ = (
        "generate_batches"
        )


def enforce_image_format(image_format):
    def decorator(function):
        K.set_image_data_format(image_format)
        return function
    return decorator

@enforce_image_format("channels_first")
def generate_batches(x_train, y_train, batch_size=1, seed=None):
    """ Infinite generator of random minibatches for a dataset.

        For general reference on (infinite) generators, see:
        https://www.python.org/dev/peps/pep-0255/

    Parameters
    ----------
    x_train: np.ndarray (N, W, H, C)
        Training images to group into minibatches.

        N: number of images
        W: width of image
        H: height of image
        C: number of channels in image

    y_train: np.ndarray (N, 1)

        Labels for trainig images.
        N: number of images

    batch_size : int, optional
        Number of images to put into one batch.

    seed: int, optional
        Random seed to use during batch generation.
        Defaults to `None`.

    Yields
    -------
    image_batch: np.ndarray (X, X)
        Batches of images of batch size `batch_size`.

    Examples
    -------
    Simple batch extraction example:
    >>> import numpy as np
    >>> from keras.datasets import cifar10
    >>> N, C , H, W = 100, 3, 32, 32  # 100 RGB images with 32 X 32 pixels
    >>> (x_train, y_train), _ = cifar10.load_data()
    >>> batch_size = 20
    >>> gen = generate_batches(x_train, y_train, batch_size)
    >>> batch = next(gen)  # extract a batch
    >>> x_batch, y_batch = batch
    >>> x_batch.shape, y_batch.shape
    ((20, 3, 32, 32), (20, 1))

    Batch extraction resizes batch size if dataset is too small:
    >>> import numpy as np
    >>> N, C , H, W = 100, 3, 32, 32  # 100 RGB images with 32 X 32 pixels
    >>> from keras.datasets import cifar10
    >>> (x_train, y_train), _ = cifar10.load_data()
    >>> x_train, y_train = x_train[10,:,:,:], y_train[10]
    >>> x_train.shape, y_train.shape
    ((10, 3, 32, 32), (10, 1))
    >>> gen = generate_batches(x_train, y_train, batch_size=20)
    >>> batch = next(gen)  # extract a batch
    >>> x_batch, y_batch = batch
    >>> x_batch.shape, y_batch.shape
    ((10, 3, 32, 32), (10,1 ))

    In this case, the batch contains exactly all datapoints:
    >>> np.allclose(x_batch, x_train), np.allclose(y_batch, y_train)
    True, True

    """
    # sanity check input
    assert(batch_size > 0)
    assert(len(x_train.shape) == 4), "Input to batch generation must be 4D array: (N_IMAGES, N_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)"

    if seed is not None:
        np.random.seed(seed)

    n_examples = x_train.shape[0]

    initial_batch_size = batch_size

    batch_size = min(initial_batch_size, n_examples)

    if initial_batch_size != batch_size:
        print("Not enough datapoints to form a minibatch. "
              "Batchsize was set to {}".format(batch_size))

    start = 0
    while True:
        if start > n_examples - batch_size:
            return
        minibatch_x = x_train[start: start + batch_size]
        minibatch_y = y_train[start: start + batch_size]
        start += batch_size
        yield minibatch_x, minibatch_y
