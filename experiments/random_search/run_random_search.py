#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-

import sys
import os
import argparse
import json

from os.path import join as path_join, dirname, realpath, abspath


from keras.datasets import mnist, cifar10

PARENT_DIRECTORY = path_join(dirname(realpath(__file__)), "..", "..")
sys.path.insert(0, PARENT_DIRECTORY)

from autoda.data_augmentation import ImageAugmentation
from experiments.benchmarks.lenet_random_search_benchmark import lenet_function


def main():

    parser = argparse.ArgumentParser(description='Simple python script to run experiments on augmented data using random search')

    parser.add_argument(
        "--benchmark", help="Neural network to be trained with augmented data"
    )
    parser.add_argument(
        "--num_epochs", default=12, help="Number of epochs", type=int
    )
    parser.add_argument(
        "--batch_size", default=128, help="Size of a mini batch", type=int
    )
    parser.add_argument(
        "--augment", default=True, help="If the data should be augmented", type=bool
    )
    parser.add_argument(
        "--dataset", help="Dataset to train neural network on"
    )
    parser.add_argument(
        "--run_id", help="The id of single job"
    )

    args = parser.parse_args()

    objective_function = {
        "LeNet": lenet_function,
    }[args.benchmark]

    dataset = {
        "mnist": mnist,
        "cifar10": cifar10
    }[args.dataset]

    path = path_join(abspath("."), "Workspace/MastersThesis/AutoDA/experiments/random_search/results", args.dataset)

    num_epochs, batch_size, augment = int(args.num_epochs), int(args.batch_size), args.augment

    sample_config = ImageAugmentation.get_config_space().sample_configuration()  # seed=123

    results = objective_function(
        sample_config=sample_config, dataset=dataset, max_epochs=num_epochs,
        batch_size=batch_size, augment=augment
    )

    path = path_join(abspath("."), "Workspace/MastersThesis/AutoDA/experiments/random_search/results", args.dataset)

    with open(os.path.join(path, "random_search_%d.json" % int(args.run_id)), "w") as fh:
        json.dump(results, fh)


if __name__ == "__main__":
    main()
