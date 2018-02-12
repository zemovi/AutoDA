#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-

import ConfigSpace as CS
from autoda.generate_batches import generate_batches
from autoda.networks.utils import normalize
from imgaug import augmenters as iaa


"""
List of Augmentation considered:
    * PadAndCrop
    * Vertical Flip
    * Horizontal Flip
    * Rotation
    * Scale
    * CoarseDropout

"""


class ImageAugmentation(object):
    """
    Creates a configuration space for hyperparameter optimization of
    a data augmentation sequence.

    Parameters
    ----------
    config:
        : augment_probability : apply whole augmentation sequence to images with augment_probability

        * PadAndCrop
        - pad_crop_probability :  apply padding and cropping to images with pad_crop_probability
        - pad_crop_lower :
        - pad_crop_upper :

        * Rotation
            - rotation_probability
            - rotation_lower
            - rotation_upper

        * Scale
            - scale_probability
            - scale_lower
            - scale_upper
        *  horizontal_flip : flip horizontally images with given probability
        *  vertical_flip : flip vertically images with given probability
        * CoarseDropout : Drop p% of all pixels by converting them to black pixels,
                          on a lower-resolution version of the image
            - coarse_dropout_size_percent : of the original size, leading to pxp squares being dropped

            - coarse_dropout_per_channel : Drop pixels independently per channel.

        * AdditiveGaussianNoise: Add white noise to images sampled per pixel from a normal distribution N(0,std).
            - gaussian_noise_scale : specifies the range the std of normal distribution is sampled from.

        * ElasticTransformation: Distort images locally by moving individual pixels around
            - elastic_transform_alpha : The strength of the movement is sampled per pixel from alpha
            - elastic_transform_sigma : Strength of distortions field
    seq :

    Examples
    -------
    >>> aug = iaa.AdditiveGaussianNoise(scale=0.1*255)
    adds gaussian noise from the distribution N(0, 0.1*255) to images
    >>> aug = iaa.CropAndPad(percent(-0.1, 0.1))


    """

    def __init__(self, config):

        self.config = config

        self.seq = iaa.Sometimes(
            self.config["augment_probability"],
            iaa.Sequential([
                iaa.Sometimes(
                    self.config["pad_crop_probability"],
                    iaa.CropAndPad(
                        percent=(
                            self.config["pad_crop_lower"],
                            self.config["pad_crop_upper"]
                        )
                    )
                ),

                iaa.Flipud(self.config["vertical_flip"]),
                iaa.Fliplr(self.config["horizontal_flip"]),
                iaa.Sometimes(
                    self.config["rotation_probability"],
                    iaa.Affine(
                        rotate=(
                            self.config["rotation_lower"],
                            self.config["rotation_upper"]
                        )
                    )
                ),

                iaa.Sometimes(
                    self.config["scale_probability"],
                    iaa.Affine(
                        scale={
                            "x": (self.config["scale_lower"], self.config["scale_upper"]),
                            "y": (self.config["scale_lower"], self.config["scale_upper"])}
                    )
                ),


                iaa.Sometimes(
                    self.config["coarse_dropout_probability"],
                    iaa.CoarseDropout(
                        p=(
                            self.config["coarse_dropout_lower"],
                            self.config["coarse_dropout_upper"]
                        ),
                        size_percent=(
                            self.config["coarse_dropout_size_percent_lower"],
                            self.config["coarse_dropout_size_percent_upper"]
                        ),

                    )
                )

            ], random_order=False
            )
        )

    from collections import namedtuple
    ParameterRange = namedtuple("ParameterRange", ["lower", "default", "upper"])

    @staticmethod
    def get_config_space(
            augment_probability=ParameterRange(lower=0., default=1., upper=1.),
            pad_crop_probability=ParameterRange(lower=0., default=1., upper=1.),
            pad_crop_lower=ParameterRange(lower=-0.25, default=-0.125, upper=0),
            pad_crop_upper=ParameterRange(lower=0., default=0.125, upper=0.25),
            rotation_probability=ParameterRange(lower=0, default=0, upper=1),
            rotation_lower=ParameterRange(lower=-180, default=0, upper=0),
            rotation_upper=ParameterRange(lower=0, default=0, upper=180),
            scale_probability=ParameterRange(lower=0., default=0., upper=1),
            scale_lower=ParameterRange(lower=0.5, default=1., upper=1.),
            scale_upper=ParameterRange(lower=1., default=1., upper=2.),
            horizontal_flip=ParameterRange(lower=0., default=0.5, upper=1),
            vertical_flip=ParameterRange(lower=0., default=0., upper=1),
            coarse_dropout_probability=ParameterRange(lower=0, default=0.0, upper=1),
            coarse_dropout_lower=ParameterRange(lower=0.0, default=0.0, upper=0.1),
            coarse_dropout_upper=ParameterRange(lower=0.1, default=0.1, upper=0.2),
            coarse_dropout_size_percent_lower=ParameterRange(lower=0.01, default=0.01, upper=0.10),
            coarse_dropout_size_percent_upper=ParameterRange(lower=0.1, default=0.1, upper=0.2),
            seed=None):

        config_space = CS.ConfigurationSpace(seed)

        hyperparameters = (
            CS.UniformFloatHyperparameter(
                "augment_probability",
                lower=augment_probability.lower,
                default_value=augment_probability.default,
                upper=augment_probability.upper,
            ),

            CS.UniformFloatHyperparameter(
                "pad_crop_probability",
                lower=pad_crop_probability.lower,
                default_value=pad_crop_probability.default,
                upper=pad_crop_probability.upper,
            ),

            CS.UniformFloatHyperparameter(
                "pad_crop_lower",
                lower=pad_crop_lower.lower,
                default_value=pad_crop_lower.default,
                upper=pad_crop_lower.upper,
            ),

            CS.UniformFloatHyperparameter(
                "pad_crop_upper",
                lower=pad_crop_upper.lower,
                default_value=pad_crop_upper.default,
                upper=pad_crop_upper.upper,
            ),

            CS.UniformFloatHyperparameter(
                "rotation_probability",
                lower=rotation_probability.lower,
                default_value=rotation_probability.default,
                upper=rotation_probability.upper,
            ),

            CS.UniformIntegerHyperparameter(
                "rotation_lower",
                lower=rotation_lower.lower,
                default_value=rotation_lower.default,
                upper=rotation_lower.upper,
            ),

            CS.UniformIntegerHyperparameter(
                "rotation_upper",
                lower=rotation_upper.lower,
                default_value=rotation_upper.default,
                upper=rotation_upper.upper,
            ),

            CS.UniformFloatHyperparameter(
                'horizontal_flip',
                lower=horizontal_flip.lower,
                default_value=horizontal_flip.default,
                upper=horizontal_flip.upper
            ),

            CS.UniformFloatHyperparameter(
                'vertical_flip',
                lower=vertical_flip.lower,
                default_value=vertical_flip.default,
                upper=vertical_flip.upper
            ),

            CS.UniformFloatHyperparameter(
                "scale_probability",
                lower=scale_probability.lower,
                default_value=scale_probability.default,
                upper=scale_probability.upper,
            ),

            CS.UniformFloatHyperparameter(
                "scale_lower",
                lower=scale_lower.lower,
                default_value=scale_lower.default,
                upper=scale_lower.upper,
            ),
            CS.UniformFloatHyperparameter(
                "scale_upper",
                lower=scale_upper.lower,
                default_value=scale_upper.default,
                upper=scale_upper.upper,
            ),

            CS.UniformFloatHyperparameter(
                "coarse_dropout_probability",
                lower=coarse_dropout_probability.lower,
                default_value=coarse_dropout_probability.default,
                upper=coarse_dropout_probability.upper,
            ),

            CS.UniformFloatHyperparameter(
                "coarse_dropout_lower",
                lower=coarse_dropout_lower.lower,
                default_value=coarse_dropout_lower.default,
                upper=coarse_dropout_lower.upper,
            ),

            CS.UniformFloatHyperparameter(
                "coarse_dropout_upper",
                lower=coarse_dropout_upper.lower,
                default_value=coarse_dropout_upper.default,
                upper=coarse_dropout_upper.upper,
            ),

            CS.UniformFloatHyperparameter(
                "coarse_dropout_size_percent_lower",
                lower=coarse_dropout_size_percent_lower.lower,
                default_value=coarse_dropout_size_percent_lower.default,
                upper=coarse_dropout_size_percent_lower.upper,
            ),

            CS.UniformFloatHyperparameter(
                "coarse_dropout_size_percent_upper",
                lower=coarse_dropout_size_percent_upper.lower,
                default_value=coarse_dropout_size_percent_upper.default,
                upper=coarse_dropout_size_percent_upper.upper,
            ),
        )

        config_space.add_hyperparameters(hyperparameters)
        return config_space

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

            aug_batch_x = self.seq.augment_images(batch_x)

            if mean is not None and variance is not None:
                aug_batch_x = normalize(aug_batch_x, mean, variance)

            yield aug_batch_x, batch_y
