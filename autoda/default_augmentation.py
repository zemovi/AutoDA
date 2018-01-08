#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-

import numpy as np
from autoda.generate_batches import generate_batches
from autoda.networks.utils import normalize
# from autoda.preprocessing import iterate_minibatches
from imgaug import augmenters as iaa


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


def apply_transform(self, x_train, y_train, mean=None, variance=None, batch_size=1):
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

    for batch_x, batch_y in generate_batches(x_train, y_train, batch_size=batch_size):

        if self.pad_width != 0 or self.pad_height != 0:
            padded_batch = pad(batch_x, self.pad_height, self.pad_width)
            cropped_batch = crop(padded_batch, crop_width=x_train.shape[2], crop_height=x_train.shape[3])  # n_crops is by default 1
        else:
            # do not crop or pad if pad_width == 0 and pad_height == 0
            cropped_batch = batch_x

        default_seq = iaa.Sequential([iaa.Fliplr(0.5)], random_order=False)
        aug_batch_x = default_seq.augment_images(cropped_batch)

        if mean is not None and variance is not None:
            aug_batch_x = normalize(aug_batch_x, mean, variance)

        yield aug_batch_x, batch_y
