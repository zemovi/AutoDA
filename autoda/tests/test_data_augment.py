#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-

import pytest
import unittest
import numpy as np

import keras
from keras import backend as K

from autoda.data_augmentation import ImageAugmentation, crop, pad
from autoda.preprocessing import generate_batches

try:
    from hypothesis import given
    import hypothesis.strategies as st

except ImportError:
    hypothesis_available = False

else:
    hypothesis_available = True


@pytest.mark.skipif(not hypothesis_available, reason="Hypothesis Test Package not installed"
)
class HypothesisTest(unittest.TestCase):
    """ Base class for property-based tests that use 'hypothesis' strategies. """
    @classmethod
    def random_image_strategy(cls, keras_dataset=keras.datasets.cifar10, n_images=1):
        (images, y_train), _ = keras_dataset.load_data()
        all_data = list(zip(images, y_train))

        indices = [
            np.random.choice(len(all_data), size=n_images)
            for _ in range(len(all_data) // n_images)
        ]
        images_with_labels = [
            [all_data[i] for i in current_indices]
            for current_indices in indices
        ]

        return st.sampled_from(images_with_labels)


@pytest.mark.skipif(not hypothesis_available, reason="Package 'Hypothesis' not installed")
class TestBatchGeneration(HypothesisTest):

    """
    def setUp(self):
        self.n_images = randint(10, 1000)


        random_images = [choice(zip(cifar10_images, y_train)) for _ in range(self.n_images)]
        print(random_images.shape)

        if len(random_images) == 1:
            random_image = random_images[0]
            self.x_train = np.reshape(random_image, (1, *random_image.shape))
        else:
            self.x_train = np.asarray(random_images)

    """
    def assertBatchDimensions(self, batch, batch_size):
        print("BATCH_DIM", batch.shape)
        if batch_size == 1:
            images = batch[0]
            labels = batch[1]
        else:
            images = np.asarray([image for image, _ in batch])
            labels = np.asarray([label for _, label in batch])

        # Assert dimensions are reasonable:
        assert(len(images[0].shape, 4))
        assert(images.shape[0] == batch_size)
        assert(images.shape[1] == self.n_channels)
        assert(images.shape[2] == self.image_width)
        assert(images.shape[3] == self.image_height)

    @given(
        st.integers(min_value=1),
        HypothesisTest.random_image_strategy(n_images=np.random.randint(2, 100))
    )
    def test_valid_batch_size(self, batch_size, data):
        images = np.asarray([image for image, _ in data])
        labels = np.asarray([label for _, label in data])

        gen = generate_batches(x_train=images, y_train=labels, batch_size=batch_size)
        batch = next(gen)

        self.assertBatchDimensions(batch, min(batch_size, len(images)))

    @given(
        st.integers(max_value=0),
        HypothesisTest.random_image_strategy(n_images=np.random.randint(2, 100))
    )
    def test_batch_size_illegal_value(self, batch_size, images):
        with pytest.raises(AssertionError):
            gen = generate_batches(images, batch_size=batch_size)
            next(gen)

    """
    def test_batch_shouldnot_overlap(self, batch_size, images):
        batch_size = self.n_images // 10
        gen = generate_batches(x_train=images, batch_size=batch_size)
        batch1 = next(gen)
        batch2 = next(gen)
        self.assertFalse(np.array_equal(batch1[-1], batch2[0]))
    """


# XXX: Batch generation should wrap single image with label in a tuple
# so that outside

"""
class TestCropImages(unittest.TestCase):

    # XXX: test for crop images function

    def setUp(self):

        self.n_images = 10

        random_images = [choice(list(zip(cifar10_images, y_train))) for _ in range(self.n_images)]
        # print(random_images.shape)

        if len(random_images) == 1:
            random_image = random_images[0]
            self.image = np.reshape(random_image, (1, *random_image.shape))
        else:
            self.image = np.asarray(random_images)

        self.n_channels = 3  # channels in cifar
        self.image_width = 32  # image_width in cifar
        self.image_height = 32  # image_height in cifar


    #XXX: what to test?
    def test_crop_imagesize_equals_cropsize(self):
        pass

    def test_crop_basic_functionality(self):

        batch_size = self.n_images
        gen = generate_batches(self.image, y_train, batch_size=batch_size)
        batch = next(gen)

        for batch in generate_batches(self.image, batch_size=batch_size):
            cropped = crop(batch, n_crops=2)
        print("cropped_images", cropped.shape)



class TestPadImages(unittest.TestCase):
    # XXX: test for pad images function

    def setUp(self):

        from random import choice, randint
        # self.n_images = randint(1, 1000)
        # self.n_images = randint(1000, 10000)
        self.n_images = 100

        random_data = [choice(list(zip(cifar10_images, y_train))) for _ in range(self.n_images)]

        if len(random_data) == 1:
            random_image, self.y_train = random_data[0]
            self.x_train = np.reshape(random_image, (1, *random_image.shape))
        else:
            self.x_train = np.asarray([image for image, _ in random_data])
            self.y_train = np.asarray([label for _, label in random_data])

        self.n_channels = 3  # channels in cifar
        self.image_width = 32  # image_width in cifar
        self.image_height = 32  # image_height in cifar

    def test_pad_basic_functionality(self):
        # pad_width, pad_height = randint(0, 200), randint(0, 200)
        pad_width, pad_height = 7, 7
        print(pad_width, pad_height)
        expected_shape = (
            self.x_train.shape[0],
            self.x_train.shape[1],
            self.x_train.shape[2] + pad_height * 2,
            self.x_train.shape[3] + pad_width * 2,
        )
        batch_padded = pad(self.x_train, pad_height=pad_height, pad_width=pad_width)
        self.assertEqual(batch_padded.shape, expected_shape)


class TestDataAugmentation(unittest.TestCase):

    def setUp(self):

        from random import choice, randint
        # self.n_images = randint(1, 1000)
        # self.n_images = randint(1000, 10000)
        self.n_images = 100

        random_data = [choice(list(zip(cifar10_images, y_train))) for _ in range(self.n_images)]

        if len(random_data) == 1:
            random_image, self.y_train = random_data[0]
            self.x_train = np.reshape(random_image, (1, *random_image.shape))
        else:
            self.x_train = np.asarray([image for image, _ in random_data])
            self.y_train = np.asarray([label for _, label in random_data])

        self.n_channels = 3  # channels in cifar
        self.image_width = 32  # image_width in cifar
        self.image_height = 32  # image_height in cifar

    def test_random_apply_transform_basic(self):

        config = ImageAugmentation.get_config_space().sample_configuration()
        img_aug = ImageAugmentation(config)
        # get a list of all batches
        # list(img_aug.apply_transform(x_train=self.image, batch_size=1))
        n_images = 1000
        n_batches = 1
        from itertools import islice
        for batch in islice(img_aug.apply_transform(x_train=self.x_train, y_train=self.y_train, batch_size=100), n_batches):
            print("AUGMENTED_BATCH DIM:", batch)

    def test_specific_config_apply_transform_basic(self):

        config = ImageAugmentation.get_config_space().sample_configuration()
        config = {
            "horizontal_flip": True,
            "vertical_flip": False,
            "pad": 3,
            "rescale": 1,
            "rotation_range": 17
        }
        img_aug = ImageAugmentation(config)
        # get a list of all batches
        # list(img_aug.apply_transform(x_train=self.image, batch_size=1))
        n_images = 1000
        n_batches = 1
        from itertools import islice
        for batch in islice(img_aug.apply_transform(x_train=self.x_train, y_train=self.y_train, batch_size=100), n_batches):
            print("AUGMENTED_BATCH DIM:", batch)

"""

def main():
    unittest.main()


if __name__ == "__main__":
    main()
