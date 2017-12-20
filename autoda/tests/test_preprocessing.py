#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-

import unittest
import pytest
import numpy as np

try:
    from hypothesis import given
    from hypothesis.strategies import integers
except ImportError:
    HYPOTHESIS_INSTALLED = False
else:
    HYPOTHESIS_INSTALLED = True

from keras.datasets import cifar10
from autoda.preprocessing import generate_batches

(CIFAR10_X_TRAIN, CIFAR10_Y_TRAIN), *_ = cifar10.load_data()
CIFAR10_N_DATAPOINTS, *_ = CIFAR10_X_TRAIN.shape


class TestSimpleBatchGeneration(unittest.TestCase):
    """Test simple case for batch generation with valid inputs"""

    @pytest.mark.skipif(not HYPOTHESIS_INSTALLED, reason="Hypothesis not installed!")
    @given(integers(min_value=1, max_value=CIFAR10_N_DATAPOINTS))
    def test_batchsize_equals_n_images(self, batch_size):
        """ Test if number of images generated in a batch are of given batch size.
        Returns
        -------
        TODO

        """

        generator = generate_batches(
            CIFAR10_X_TRAIN, CIFAR10_Y_TRAIN, batch_size=batch_size
        )

        x_batch, y_batch = next(generator)
        n_images, *_ = x_batch.shape

        assert n_images == batch_size

    @pytest.mark.skipif(not HYPOTHESIS_INSTALLED, reason="Hypothesis not installed!")
    @given(integers(max_value=0))
    def test_invalid_batchsize_error(self, batch_size):
        with pytest.raises(AssertionError):
            generator = generate_batches(
                CIFAR10_X_TRAIN, CIFAR10_Y_TRAIN, batch_size=batch_size
            )
            next(generator)

    @given(integers(min_value=CIFAR10_N_DATAPOINTS))
    def test_batchsize_larger_n_images(self, batch_size):

        generator = generate_batches(
            CIFAR10_X_TRAIN, CIFAR10_Y_TRAIN, batch_size=batch_size
        )

        x_batch, y_batch = next(generator)
        n_images, *_ = x_batch.shape

        assert n_images == CIFAR10_N_DATAPOINTS

    @pytest.mark.skipif(not HYPOTHESIS_INSTALLED, reason="Hypothesis not installed!")
    @given(integers(min_value=1, max_value=10))
    def test_all_images_in_dataset(self, batch_size):
        generator = generate_batches(
            CIFAR10_X_TRAIN, CIFAR10_Y_TRAIN, batch_size=batch_size
        )

        x_batch, y_batch = next(generator)

        in_dataset = np.zeros(len(x_batch))
        last_image = 0

        for image in CIFAR10_X_TRAIN:
            if in_dataset.all():
                continue
            if np.array_equal(x_batch[last_image, ...], image):
                in_dataset[last_image] = 1
                last_image += 1
        assert in_dataset.all()


if __name__ == "__main__":
    unittest.main()
