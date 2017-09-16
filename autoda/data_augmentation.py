#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-

from abc import ABCMeta
import numpy as np
import ConfigSpace as CS

from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K


def generate_batches(x_train, y_train, batch_size=1, seed=None):
    """ Infinite generator of random minibatches for a dataset.

        For general reference on (infinite) generators, see:
        https://www.python.org/dev/peps/pep-0255/

    Parameters
    ----------
    X: np.ndarray (N, W, H, C)
        Images to group into minibatches.

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
    >>> N, C , H, W = 100, 3, 32, 32  # 100 RGB images with 32 X 32 pixels
    >>> X = np.asarray([np.random.uniform(-10, 10, D) for _ in range(N)]) # XXX Use images here instead
    >>> X.shape
    (100, 3)
    >>> batch_size = 20
    >>> gen = generate_batches(X, batch_size)
    >>> batch = next(gen)  # extract a batch
    XXX VALIDATE RESULTS HERE

    Batch extraction resizes batch size if dataset is too small:
    >>> import numpy as np
    >>> N, D = 10, 3  # 10 datapoints with 3 features each
    >>> X = np.asarray([np.random.uniform(-10, 10, D) for _ in range(N)]) # XXX Use images here instead
    >>> X.shape
    (10, 3)
    >>> batch_size = 20
    >>> gen = generate_batches(X, batch_size)
    >>> batch = next(gen)  # extract a batch XXX Ensure that batch size is now 10, e.g. by asserting batch shape
    XXX VALIDATE RESULTS HERE

    In this case, the batch contains exactly all datapoints:
    >>> np.allclose(batch, X)
    True

    """
    assert(batch_size > 0)
    assert(len(x_train.shape) == 4), "Input to batch generation must be 4D array: (N_IMAGES, N_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)"

    if seed is not None:
        np.random.seed(seed)

    if K.image_data_format() == 'channels_last':
        K.set_image_data_format('channels_first')
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


def crop(batch, crop_width=32, crop_height=32, n_crops=1):

    """ crops :n_crops: images of size 1X32X32 from batch of images selected
        from random positions

    Parameters
    ----------
    batch : numpy.ndarray (n_images, n_channels, image_width, image_height)
        Batch of images to crop.

    crop_height: int, default=32
        Number of pixels to be cropped from vertical axis.

    crop_width: int, default=32
        Number of pixels to be padded on horizontal axis.

    n_crops: int
        Number of images to crop from single image, default=1

    Returns
    -------
    cropped_images : numpy.ndarray (n_images x n_crops, n_channels, image_height, image_width)

    """

    assert(len(batch.shape) == 4), "Input to crop must be 4D array: \
                                    (N_IMAGES, N_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)\
                                    but found"
    n_examples = batch.shape[0]
    n_channels = batch.shape[1]
    image_height = batch.shape[2]
    image_width = batch.shape[3]

    if crop_width == image_width and crop_height == image_height:
        return batch

    msg = "Crop {0}: '{1}' should be smaller than image {0}: '{2}'".format
    # assert crop and image widths
    assert(crop_width < image_width), msg("width", crop_width, image_width)
    assert crop_height < image_height, msg("height", crop_height, image_height)

    # crops a single image from batch staring at random positions(height/width)

    def crop_image(image):
        crop_start_width = np.random.randint(0, image_width - crop_width)
        crop_start_height = np.random.randint(0, image_height - crop_height)

        # crops each color RGB channel for single image
        def crop_color_channel(color_channel, crop_start_width, crop_start_height):
            return color_channel[crop_start_width: crop_start_width + crop_width,
                                 crop_start_height: crop_start_height + crop_height]
        return [crop_color_channel(color_channel, crop_start_width, crop_start_height)
                for color_channel in image]

    cropped_images = []

    for image in batch:
        # list of cropped images randomly cropped from batch of images
        cropped_images.extend([crop_image(image) for _ in range(n_crops)])

    return np.asarray(cropped_images)


def pad(batch, pad_height, pad_width):

    """ Pads each side of each image in :batch: by specified values
        :n_pixels_height: and :n_pixels_width. Padding is zero-padding
        (adds black pixels to image).

    Parameters
    ----------
    batch : numpy.ndarray (n_images, n_channels, image_width, image_height)
        Batch of images to pad.

    n_pixels_height :int
        Number of pixels to be padded on vertical axis.

    n_pixels_width : int
        Number of pixels to be padded horizontal axis.

    Returns
    -------
    batch_padded : numpy.ndarray (n_images, n_channels, TODO DOKU)
        Padded images.

    """

    assert(len(batch.shape) == 4), "Image batches should be 4D: (n_images, n_channels, image_width, image_height)"

    pads = (pad_height, pad_height), (pad_width, pad_width)

    def pad_by_zero(color_channel):
        return np.pad(color_channel, pads, "constant", constant_values=0)

    batch_padded = []
    for image in batch:
        image_padded = np.asarray([pad_by_zero(color_channel) for color_channel in image])

        batch_padded.append(image_padded)

    return np.asarray(batch_padded)


def enforce_image_format(image_format):
    def decorator(function):
        K.set_image_data_format(image_format)
        return function
    return decorator


@enforce_image_format("channels_first")
class ImageAugmentation(object):
    """Data augmentation for image data. """
    def __init__(self, config):

        self.config = config

        self.pad_height = self.config["pad"]
        self.pad_width = self.config["pad"]
        self.datagen = ImageDataGenerator(rotation_range=self.config["rotation_range"],
                                          rescale=self.config["rescale"],
                                          horizontal_flip=self.config["horizontal_flip"],
                                          vertical_flip=self.config["vertical_flip"])

    @staticmethod
    def get_config_space(
            rotation_range=[0, 0, 180],
            rescale=[0, 0, 2],
            pad=None,
            horizontal_flip=[True, False],
            vertical_flip=[True, False],
            seed=None):

        cs = CS.ConfigurationSpace(seed)

        HPs = [CS.UniformIntegerHyperparameter("rotation_range", lower=rotation_range[0], upper=rotation_range[2], default=rotation_range[1]),
               CS.UniformIntegerHyperparameter("rescale", lower=rescale[0], upper=rescale[2], default=rescale[1]),
               CS.UniformIntegerHyperparameter("pad", lower=0, upper=8),
               CS.CategoricalHyperparameter('horizontal_flip', horizontal_flip),
               CS.CategoricalHyperparameter("vertical_flip", vertical_flip),
               ]

        [cs.add_hyperparameter(hp) for hp in HPs]

        return cs

    def apply_transform(self, x_train, y_train, batch_size=1):
        """  Applies image augmentation on given training samples

        Parameters
        ----------

        x_train: numpy.ndarray (n_images, n_channels, image_width, image_height)
                 Input training images to be augumented. Must be 4D

        batch_size : int, optional
                    Number of images in one batch
        Yields
        -------
        image_batch: numpy.ndarray (n_images_in_batch, n_channels, image_width, image_height)
                     Batches of augmented images
        """

        for batch, y_train in generate_batches(x_train, y_train, batch_size=batch_size):

            if self.pad_width != 0 or self.pad_height != 0:
                padded_batch = pad(batch, self.pad_height, self.pad_width)
                cropped_batch = crop(padded_batch, crop_width=x_train.shape[2], crop_height=x_train.shape[3])  # n_crops is by default 1
            else:
                # do not crop or pad if pad_width == 0 and pad_height == 0
                cropped_batch = batch

            # apply ImageDataGenerator instance on (padded and cropped) sample
            self.datagen.fit(cropped_batch)

            yield from self.datagen.flow(cropped_batch, y_train, batch_size=batch_size)
