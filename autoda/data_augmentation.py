#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-

import ConfigSpace as CS
from autoda.generate_batches import generate_batches
from autoda.networks.utils import normalize
# from autoda.preprocessing import iterate_minibatches
from imgaug import augmenters as iaa


class ImageAugmentation(object):
    """Data augmentation for image data.

    Parameters
    ----------

    Examples
    -------

    """

    def __init__(self, config):

        self.config = config

        self.seq = iaa.Sometimes(self.config["augment_probability"],
                iaa.Sequential([
                    iaa.Sometimes(self.config["pad_probability"],
                                  iaa.Pad(
                                      percent=(
                                          self.config["pad_lower"],
                                          self.config["pad_upper"]
                                          )
                                      )
                                  ),

                    iaa.Sometimes(self.config["crop_probability"],
                                  iaa.Crop(
                                      percent=(
                                          self.config["crop_lower"],
                                          self.config["crop_upper"]
                                          )
                                      )
                                  ),
                    iaa.Flipud(self.config["vertical_flip"]),
                    iaa.Fliplr(self.config["horizontal_flip"]),
                    iaa.Sometimes(self.config["rotation_probability"],
                                  iaa.Affine(
                                      rotate=(
                                          self.config["rotation_lower"],
                                          self.config["rotation_upper"]
                                          )
                                      )
                                  ),
                    iaa.Sometimes(self.config["scale_probability"],
                                  iaa.Affine(
                                      scale={"x": (self.config["scale_lower"], self.config["scale_upper"]),
                                             "y": (self.config["scale_lower"], self.config["scale_upper"])}
                                      )
                                  )
                ], random_order=False
            )
        )

    from collections import namedtuple
    ParameterRange = namedtuple("ParameterRange", ["lower", "default", "upper"])

    @staticmethod
    def get_config_space(

            augment_probability=ParameterRange(lower=0, default=0.5, upper=1),
            rotation_lower=ParameterRange(lower=-180, default=0, upper=0),
            rotation_upper=ParameterRange(lower=0, default=0, upper=180),
            rotation_probability=ParameterRange(lower=0, default=0, upper=1),
            scale_lower=ParameterRange(lower=0.5, default=1., upper=1.),
            scale_upper=ParameterRange(lower=1., default=1., upper=2.),
            scale_probability=ParameterRange(lower=0, default=0, upper=1),
            horizontal_flip=ParameterRange(lower=0, default=0, upper=1),
            vertical_flip=ParameterRange(lower=0, default=0, upper=1),
            crop_lower=ParameterRange(lower=0, default=0, upper=0.1),
            crop_upper=ParameterRange(lower=0.1, default=0.1, upper=0.3),
            crop_probability=ParameterRange(lower=0, default=0, upper=1),
            pad_lower=ParameterRange(lower=0, default=0, upper=0.1),
            pad_upper=ParameterRange(lower=0.1, default=0.1, upper=0.3),
            pad_probability=ParameterRange(lower=0, default=0, upper=1),
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

            CS.UniformFloatHyperparameter(
                'vertical_flip',
                lower=vertical_flip.lower,
                default=vertical_flip.default,
                upper=vertical_flip.upper
            ),

            CS.UniformFloatHyperparameter(
                "scale_lower",
                lower=scale_lower.lower,
                default=scale_lower.default,
                upper=scale_lower.upper,
            ),
            CS.UniformFloatHyperparameter(
                "scale_upper",
                lower=scale_upper.lower,
                default=scale_upper.default,
                upper=scale_upper.upper,
            ),

            CS.UniformFloatHyperparameter(
                "scale_probability",
                lower=scale_probability.lower,
                default=scale_probability.default,
                upper=scale_probability.upper,
            ),

            CS.UniformFloatHyperparameter(
                "crop_lower",
                lower=crop_lower.lower,
                default=crop_lower.default,
                upper=crop_lower.upper,
            ),

            CS.UniformFloatHyperparameter(
                "crop_upper",
                lower=crop_upper.lower,
                default=crop_upper.default,
                upper=crop_upper.upper,
            ),

            CS.UniformFloatHyperparameter(
                "crop_probability",
                lower=crop_probability.lower,
                default=crop_probability.default,
                upper=crop_probability.upper,
            ),

            CS.UniformFloatHyperparameter(
                "pad_lower",
                lower=pad_lower.lower,
                default=pad_lower.default,
                upper=pad_lower.upper,
            ),

            CS.UniformFloatHyperparameter(
                "pad_upper",
                lower=pad_upper.lower,
                default=pad_upper.default,
                upper=pad_upper.upper,
            ),

            CS.UniformFloatHyperparameter(
                "pad_probability",
                lower=pad_probability.lower,
                default=pad_probability.default,
                upper=pad_probability.upper,
            ),
            CS.UniformFloatHyperparameter(
                "augment_probability",
                lower=augment_probability.lower,
                default=augment_probability.default,
                upper=augment_probability.upper,
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

        # for batch_x, batch_y in iterate_minibatches(inputs=x_train, targets=y_train, batchsize=batch_size):
        for batch_x, batch_y in generate_batches(x_train, y_train, batch_size=batch_size):

            aug_batch_x = self.seq.augment_images(batch_x)

            if mean is not None and variance is not None:
                aug_batch_x = normalize(aug_batch_x, mean, variance)

            yield aug_batch_x, batch_y


