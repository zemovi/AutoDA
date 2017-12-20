
from keras import backend as K

import numpy as np
from collections import defaultdict


def enforce_image_format(image_format):
    def decorator(function):
        K.set_image_data_format(image_format)
        return function
    return decorator


@enforce_image_format("channels_last")
def get_data(dataset):

    # The data, shuffled and split between train and test sets:
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    return x_train, y_train, x_test, y_test


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


def get_num_classes(y_train):
    import numpy as np
    num_classes = int(np.max(y_train)) + 1

    try:
        num_elements, num_classes = len(num_classes), num_classes[0]
    except TypeError:
        return num_classes
    else:
        assert num_elements == 1
        return num_classes


def get_input_shape(x_train):
    print(x_train.shape)
    _, *shape = x_train.shape
    return shape


def compute_zero_mean_unit_variance(x, mean=None, std=None):

    """ Compute zero mean and unit variance of training set
        for normalizating augmented batch in training

    """

    if mean is None:
        mean = np.mean(x, axis=0)
    if std is None:
        std = np.std(x, axis=0)

    return mean, std


def normalize(x, mean, std):
    """ Normalize data by its zero mean and standard variance

    """

    x_normalized = (x - mean) / (std + 1e-7)

    return x_normalized
