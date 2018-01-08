#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-

import sys
import os
import argparse
import json

from os.path import join as path_join, abspath


from keras.datasets import mnist, cifar10
sys.path.insert(0, abspath(path_join(__file__, "..", "..")))

from autoda.data_augmentation import ImageAugmentation
from autoda.networks.train import objective_function


def main():

    parser = argparse.ArgumentParser(description='Simple python script to run experiments on augmented data using random search')

    parser.add_argument(
        "--benchmark", help="Neural network to be trained with augmented data"
    )
    parser.add_argument(
        "--max_epochs", default=12, help="Maximum number of epochs to train network", type=int
    )
    parser.add_argument(
        "--batch_size", default=128, help="Size of a mini batch", type=int
    )
    parser.add_argument(
        "--augment", default=False, help="If the data should be augmented"
    )
    parser.add_argument(
        "--dataset", help="Dataset to train neural network on"
    )
    parser.add_argument(
        "--run_id", help="The id of single job"
    )

    args = parser.parse_args()

    benchmark = args.benchmark

    dataset = {
        "mnist": mnist,
        "cifar10": cifar10
    }[args.dataset]

    max_epochs, batch_size, augment = int(args.max_epochs), int(args.batch_size), args.augment

    sample_config = None

    if augment:
        sample_config = ImageAugmentation.get_config_space().sample_configuration()  # seed=123

    results = objective_function(
        configuration=sample_config, dataset=dataset, benchmark=benchmark, max_epochs=max_epochs,
        batch_size=batch_size, time_budget=900
    )

    path = path_join(abspath("."), "AutoData", args.dataset)

    with open(os.path.join(path, "random_search_{}_{}.json".format(args.dataset, int(args.run_id))), "w") as fh:
        json.dump(results, fh)


if __name__ == "__main__":
    main()
