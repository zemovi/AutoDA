#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-


import time

import keras
import keras.backend.tensorflow_backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization



from sklearn.model_selection import train_test_split

from autoda.data_augmentation import ImageAugmentation
from autoda.preprocessing import enforce_image_format
from autoda.util.normalization import compute_zero_mean_unit_variance, normalize

from collections import defaultdict

import logging
logging.basicConfig(level=logging.INFO)


def _merge_dict(dict_list):
    dd = defaultdict(list)
    for d in dict_list:
        for key, value in d.items():
            if not hasattr(value, '__iter__'):
                value = (value,)
            [dd[key].append(v) for v in value]
    return dict(dd)


def _update_history(train_history, history):
    if len(train_history) == 0:
        train_history = history
    else:
        train_history = _merge_dict([train_history, history])
    return train_history


@enforce_image_format("channels_last")
def get_data(dataset):

    # The data, shuffled and split between train and test sets:
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    return x_train, y_train, x_test, y_test


@enforce_image_format("channels_last")
def preprocess_data(x_train, y_train, x_test, y_test, augment):

    num_classes = 10

    img_rows, img_cols = x_train.shape[2], x_train.shape[2]

    # reshape 3D image to 4D
    if x_train.ndim == 3:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    # compute zero mean and unit variance for normalization
    mean, variance = compute_zero_mean_unit_variance(x_train)

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2)

    # normalize validation and test set, training set is normalized after augmentation
    # using the mean and variance computed from non-augmented data
    # since augmentated data is sampled from the same distribution as non-augmented data
    # approximately the same mean and variance

    if not augment:
        print("normalize training set beforehand if no data_augmentation")
        x_train = normalize(x_train, mean, variance)
    x_valid = normalize(x_valid, mean, variance)
    x_test = normalize(x_test, mean, variance)

    # reshape vector from (N,1) to (N,)
    y_train = y_train.reshape((y_train.shape[0], ))
    y_valid = y_valid.reshape((y_valid.shape[0], ))

    # dimensions of data
    print(x_train.shape, 'x_train Dimensions')
    print(x_train.shape[0], 'train samples')
    print(x_valid.shape[0], 'validation samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_valid = keras.utils.to_categorical(y_valid, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_valid, y_valid, x_test, y_test, mean, variance


def alexnet_function(sample_config, dataset, max_epochs, batch_size, augment):

    # load data
    x_train, y_train, x_test, y_test = get_data(dataset)

    # preprocess data
    x_train, y_train, x_valid, y_valid, x_test, y_test, mean, variance = preprocess_data(x_train, y_train, x_test, y_test, augment)

    num_classes = 10
    print(y_train.shape, x_train.shape)

    print("number of classes", num_classes)
    num_epochs = 0
    time_budget = 2000
    used_budget = 0.
    duration_last_epoch = 0.
    runtime = []
    train_history = {}

    with K.tf.device('/gpu:1'):

        K.set_session(K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)))

        # AlexNet
        model = Sequential()
        model.add(Conv2D(140, (5, 5), padding='same',
                         input_shape=x_train.shape[1:]))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
        model.add(Conv2D(94, (5, 5)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

        model.add(Conv2D(126, (5, 5), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

        model.add(Flatten())
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

        opt = keras.optimizers.Adam(lr=0.0016681005372000575)
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=opt,
                      metrics=['accuracy'])

        start_time = time.time()
        while(num_epochs < max_epochs) and \
                (used_budget + 1.11 * duration_last_epoch < time_budget):
            if not augment:
                print('Not using data augmentation.')

                history = model.fit(x_train, y_train,
                                    batch_size=batch_size,
                                    epochs=num_epochs + 1,
                                    validation_data=(x_valid, y_valid),
                                    initial_epoch=num_epochs,
                                    shuffle=True)

                train_history = _update_history(train_history, history.history)
            else:
                print('Using real-time data augmentation.')

                augmenter = ImageAugmentation(sample_config)

                # Fit the model on the batches augmented data generated by apply transform
                history = model.fit_generator(augmenter.apply_transform(x_train, y_train, mean, variance,
                                                                        batch_size=batch_size),

                                              steps_per_epoch=x_train.shape[0] // batch_size,
                                              epochs=num_epochs + 1,
                                              validation_data=(x_valid, y_valid),
                                              initial_epoch=num_epochs
                                              )
                train_history = _update_history(train_history, history.history)
            num_epochs += len(history.history.get("loss", []))
            duration_last_epoch = (time.time() - start_time) - used_budget
            used_budget += duration_last_epoch
            print("used_budget", used_budget, "duration_last_epoch", duration_last_epoch, "time_budget", time_budget)
            runtime.append(time.time() - start_time)

        # Evaluate model with test data set and share sample prediction results
        score = model.evaluate(x_valid, y_valid, verbose=0)

    result = dict()
    if not augment:
        result["configs"] = {}
    else:
        result["configs"] = sample_config.get_dictionary()
    result["train_accuracy"] = history.history['acc'][-1]
    result["validation_error"] = 1 - score[1]
    result["used_budget"] = used_budget
    result["train_history"] = train_history
    result["augment"] = augment

    return result
