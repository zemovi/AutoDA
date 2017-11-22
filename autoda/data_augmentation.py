#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-

import time
import numpy as np
import ConfigSpace as CS

from keras.preprocessing.image import ImageDataGenerator

from autoda.preprocessing import generate_batches, enforce_image_format


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



class ImgAugDataGenerator(object):
    """ Wrapper over imgaug to simulate the ImageDataGenerator of keras. """
    def __init__(self, crop_and_pad, rotation_lower, rotation_upper,
            rotation_proability):
        # set up parameters for img aug here
        self.seq = iaa.Sequential(
                [
                    iaa.Sometimes(self.config["rotation_probability"])(
                        iaa.Affine(
                            rotate=(
                                self.config["rotation_lower"],
                                self.config["rotation_upper"]
                            ),
                        )
                    ),

                ], random_order=True
        )


    def flow(self, x_batch, y_batch, batch_size=32):
        """ yield batches of augmented data of size batch size and their respective
            labels
        """
            aug_x_batch = self.seq.augment_images(x_batch)
            # TODO:  how to assign the corresponding label to the augmented image
            # aug_y_batch =


        yield aug_x_batch



@enforce_image_format("channels_first")
class ImageAugmentation(object):
    """Data augmentation for image data. """
    def __init__(self, config, method="keras"):

        self.config = config

        self.pad_height = self.config["pad"]
        self.pad_width = self.config["pad"]
        if method == "keras":
            self.datagen = ImageDataGenerator(rotation_range=self.config["rotation_range"],
                                              rescale=self.config["rescale"],
                                              horizontal_flip=self.config["horizontal_flip"],
                                              vertical_flip=self.config["vertical_flip"])
        else:
            self.datagen = ImgAugDataGenerator(
                crop_and_pad=self.config(["crop_and_pad"],
                rotation_lower = self.config(["rotation_lower"]),
                rotation_upper = self.config(["rotation_upper"]),
                rotation_proability = self.config(["rotation_proability"])
            )

    from collections import namedtuple
    ParameterRange = namedtuple("ParameterRange", ["lower", "default", "upper"])

    @staticmethod
    def get_config_space(
            method="keras",
            keras_rotation_range=ParameterRange(lower=0, default=0, upper=180),
            keras_rescale=ParameterRange(lower=0.5, default=1.0, upper=2.0),
            pad=ParameterRange(lower=0, default=0, upper=8),
            keras_horizontal_flip_default=True,
            keras_vertical_flip_default=True,
            seed=None):

        config_space = CS.ConfigurationSpace(seed)

        if method == "keras":
            hyperparameters = (
                CS.UniformIntegerHyperparameter(
                    "rotation_range",
                    lower=keras_rotation_range.lower,
                    default=keras_rotation_range.default,
                    upper=keras_rotation_range.upper
                ),
                CS.UniformFloatHyperparameter(
                    "rescale",
                    lower=keras_rescale.lower,
                    default=keras_rescale.default,
                    upper=keras_rescale.upper
                ),
                CS.UniformIntegerHyperparameter(
                    "pad",
                    lower=pad.lower,
                    default=pad.default,
                    upper=pad.upper
                ),
                CS.CategoricalHyperparameter(
                    'horizontal_flip',
                    choices=(True, False),
                    default=keras_horizontal_flip_default
                ),
                CS.CategoricalHyperparameter(
                    'vertical_flip',
                    choices=(True, False),
                    default=keras_vertical_flip_default
                )
            )
        elif method == "imgaug":
            # XXX: imgaug config space parameters here
            hyperparameters = (
                CS.UniformIntegerHyperparameter(
                    "rotation_lower",
                    lower=imgaug_rotation_lower.lower,
                    default=imgaug_rotation_lower.default,
                    upper=imgaug_rotation_lower.upper,
                ),
                CS.UniformIntegerHyperparameter(
                    "rotation_upper",
                    lower=imgaug_rotation_upper.lower,
                    default=imgaug_rotation_upper.default,
                    upper=imgaug_rotation_upper.upper,
                ),

                CS.UniformFloatHyperparameter(
                    "rotation_probability",
                    lower=imgaug_rotation_probability.lower,
                    default=imgaug_rotation_probability.default,
                    upper=imgaug_rotation_probability.upper,
                ),
            )
        else:
            raise ValueError(
                "Augmentation method '{}' unsupported!".format(method)
            )

        config_space.add_hyperparameters(hyperparameters)

        return config_space

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
                pad_start_time = time.time()
                padded_batch = pad(batch, self.pad_height, self.pad_width)
                crop_start_time = time.time()
                cropped_batch = crop(padded_batch, crop_width=x_train.shape[2], crop_height=x_train.shape[3])  # n_crops is by default 1
            else:
                # do not crop or pad if pad_width == 0 and pad_height == 0
                cropped_batch = batch
            # apply ImageDataGenerator instance on (padded and cropped) sample

            yield from self.datagen.flow(cropped_batch, y_train, batch_size=batch_size)
