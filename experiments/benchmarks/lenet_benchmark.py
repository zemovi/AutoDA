#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-


import time
# import signal

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10

from sklearn.model_selection import train_test_split

from autoda.data_augmentation import ImageAugmentation
from autoda.preprocessing import enforce_image_format

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

    num_classes = 10

    # input image dimensions

    # The data, shuffled and split between train and test sets:
    (X, y), (x_test, y_test) = dataset.load_data()

    img_rows, img_cols = X.shape[2], X.shape[2]

    # reshape 3D image to 4D
    if X.ndim == 3:
        print("HERE")
        X = X.reshape(X.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    # X = X.astype('float32')
    # x_test = x_test.astype('float32')

    # X /= 255
    # x_test /= 255

    x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
    print("X", X.shape)

    print('x_train shape:', x_train.shape)

    print(x_train.shape[0], 'train samples')
    print(x_valid.shape[0], 'validation samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_valid = keras.utils.to_categorical(y_valid, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_valid, y_valid, x_test, y_test, img_rows, img_cols


def lenet_function(sample_config, dataset, max_epochs, batch_size, augment):

    """
    config = ImageAugmentation.get_config_space().sample_configuration()
    config['augment_probability'] = 0.0
    # x_train, y_train, x_valid, y_valid, _, _, img_rows, img_cols = get_data(dataset)
    (x_train, y_train), _ = cifar10.load_data()
    augmenter = ImageAugmentation(config)
    print("SAMPLE_CONFIG", config)

    augment_x_train, augment_y_train = next(augmenter.apply_transform(x_train, y_train, batch_size=x_train.shape[0]))
    print("AUG_1", x_train[0], "y_train", y_train[0], "aug_x", augment_x_train[0], "aug_y", augment_y_train[0])

    augment_x_train = augment_x_train.astype('float32')

    # augment_x_train /= 255

    augment_y_train = keras.utils.to_categorical(augment_y_train, 10)
    print("AUG_2", x_train[0], "y_train", y_train[0], "aug_x", augment_x_train[0], "aug_y", augment_y_train[0])

    # x, x_val y, y_val = train_test_split(augment_x_train, augment_y_train, test_size=0.2)



   # Convert class vectors to binary class matrices.

    """

    x_train, y_train, x_valid, y_valid, _, _, img_rows, img_cols = get_data(dataset)
    input_shape = (img_rows, img_cols, x_train.shape[3])  # NWHC
    print("INPUT_SHAPE", input_shape)

    num_classes = 10

    num_epochs = 0
    time_budget = 900
    used_budget = 0.
    duration_last_epoch = 0.
    runtime = []
    train_history = {}

    # LeNet
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # Let's train the model using ADAM

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    start_time = time.time()
    while(num_epochs < max_epochs) and \
            (used_budget + 1.11 * duration_last_epoch < time_budget):
        if not augment:
            print('Not using data augmentation.')

            x_train = x_train.astype('float32')
            x_train /= 255
            history = model.fit(x_train, y_train,
                                batch_size=batch_size,
                                epochs=num_epochs + 1,
                                validation_data=(x_valid, y_valid),
                                initial_epoch=num_epochs,
                                shuffle=True)

            train_history = _update_history(train_history, history.history)
        else:
            print('Using real-time data augmentation.')
#            sample_config['augment_probability'] = 0.5
#            sample_config['crop_lower'] = 0.045156496217882106
#            sample_config['crop_probability'] = 0.18876057886170727
#            sample_config['crop_upper'] = 0.1677419110680472
#            sample_config['horizontal_flip'] = 0.5
#            sample_config['pad_lower'] = 0.0902640994070404
#            sample_config['pad_probability'] =0.8728219624949363
#            sample_config['pad_upper'] = 0.1609875018382957
#            sample_config['rotation_lower'] = -20
#            sample_config['rotation_probability'] = 0.22256115473092497
#            sample_config['rotation_upper'] = 20
#            sample_config['scale_lower'] = 0.5443443266762729
#            sample_config['scale_probability'] = 0.7526542698426937
#            sample_config['scale_upper'] = 1.9682111619238962
#            sample_config['vertical_flip'] = 0.

            print("sample_config", sample_config)
            augmenter = ImageAugmentation(sample_config)

            # Fit the model on the batches generated by datagen.flow().
            history = model.fit_generator(augmenter.apply_transform(x_train, y_train,
                                                                    batch_size=batch_size),

                                          steps_per_epoch=x_train.shape[0] // batch_size,
                                          epochs=num_epochs + 1,
                                          validation_data=(x_valid, y_valid),
                                          initial_epoch=num_epochs
                                          )
            train_history = _update_history(train_history, history.history)
            print(train_history)
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
