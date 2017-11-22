#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-

import ConfigSpace as CS

from autoda.preprocessing import generate_batches
from imgaug import augmenters as iaa


class ImageAugmentation(object):
    """Data augmentation for image data. """
    def __init__(self, config):

        self.config = config

        self.seq = iaa.Sequential(
            [
                iaa.Fliplr(self.config["horizontal_flip"]),
                iaa.Sometimes(self.config["rotation_probability"],
                              iaa.Affine(
                                    rotate=(
                                        self.config["rotation_lower"],
                                        self.config["rotation_upper"]
                                    ),
                              )
                ),

            ], random_order=True
        )

    from collections import namedtuple
    ParameterRange = namedtuple("ParameterRange", ["lower", "default", "upper"])

    @staticmethod
    def get_config_space(

            rotation_lower=ParameterRange(lower=-180, default=0, upper=0),
            rotation_upper=ParameterRange(lower=0, default=0, upper=180),
            rotation_probability=ParameterRange(lower=0, default=0, upper=1),
            horizontal_flip=ParameterRange(lower=0, default=0, upper=1),
            seed=None):

        config_space = CS.ConfigurationSpace(seed)

        hyperparameters = (
            CS.UniformIntegerHyperparameter(
                "rotation_lower",
                lower=rotation_lower.lower,
                default=rotation_lower.default,
                upper=rotation_lower.upper,
            ),
            CS.UniformIntegerHyperparameter(
                "rotation_upper",
                lower=rotation_upper.lower,
                default=rotation_upper.default,
                upper=rotation_upper.upper,
            ),

            CS.UniformFloatHyperparameter(
                "rotation_probability",
                lower=rotation_probability.lower,
                default=rotation_probability.default,
                upper=rotation_probability.upper,
            ),

            CS.UniformFloatHyperparameter(
                'horizontal_flip',
                lower=horizontal_flip.lower,
                default=horizontal_flip.default,
                upper=horizontal_flip.upper
            ),
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
            yield self.seq.augment_images(batch)
