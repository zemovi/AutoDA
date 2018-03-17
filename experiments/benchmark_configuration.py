#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-
#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-

import sys
import os
import argparse
import json
from functools import partial

from os.path import join as path_join, abspath


from keras.datasets import mnist, cifar10, cifar100
sys.path.insert(0, abspath(path_join(__file__, "..", "..")))

from autoda.data_augmentation import ImageAugmentation
from autoda.networks.train import objective_function
from autoda.networks.utils import get_data


def main():

    parser = argparse.ArgumentParser(description='Simple python script to benchmark data augmentation configurations.')


    parser.add_argument(
        "--benchmark",
        default="AlexNet",
        help="Neural network to be trained with augmented data"
    )

    parser.add_argument(
        "--pipeline",
        dest="pipeline",
        default="pipeline1",
        help="Data augmentation pipeline to use, choice:{standard, pipeline1, pipeline2}"
    )

    parser.add_argument(
        "--max-epochs",
        default=200,
        dest="max_epochs",
        type=int,
        help="Maximum number of epochs to train network"
    )
    parser.add_argument(
        "--batch-size", default=128,
        dest="batch_size",
        type=int,
        help="Size of a mini batch",
    )

    parser.add_argument(
        "--augment", action="store_true", help="If the data should be augmented, if flag not set defaults to false"
    )

    parser.add_argument(
        "--time-budget", default=7200, help="Maximum time budget to train a network",
        type=int, dest="time_budget"
    )
    parser.add_argument(
        "--dataset",
        default="cifar10",
        help="Dataset to train neural network on"

    )
    parser.add_argument(
        "--optimizer",
        default="hyperband",
        help="optimizer"
    )
    parser.add_argument(
        "--run-id", help="The id of single job", dest="run_id"
    )

    parser.add_argument(
    "--output-file", "-o", help="Output File",
    default=None, dest="output_file"
    )

    parser.add_argument(
        "--configuration_file",
        help="Path to JSON file containing a configuration dictionary for "
             "our data augmentation. "
             "Defaults to `None`, which uses no data augmentation.",
        default=None, dest="configuration_file"
    )


    args = parser.parse_args()

    print(args.pipeline, args.run_id)

    default_output_file = path_join(
        abspath("."), "AutoData",
        args.dataset,
        args.optimizer,
        args.benchmark,
        "{pipeline}_{run_id}.json".format(
            pipeline=args.pipeline, run_id=int(args.run_id)
        )
    )

    args.output_file = args.output_file or default_output_file

    configuration = None

    if args.configuration_file:
        with open(args.configuration_file, "r") as configuration_file:
            configuration = json.load(configuration_file)["best_run_info"]["info"]["configs"] # XXX: rewrite json format to fit optimized results from hyperband, smac or defualt
            print("configuration_file", configuration)

    augment = args.augment or (args.configuration_file is not None)

    benchmark = args.benchmark

    dataset = {"mnist": mnist, "cifar10": cifar10, "cifar100":cifar100}[args.dataset]

    max_epochs, batch_size, time_budget = int(args.max_epochs), int(args.batch_size), int(args.time_budget)

    data = get_data(dataset, augment)

    # XXX: Remove me
    if augment:
        configuration = ImageAugmentation.get_config_space().sample_configuration()


    results = objective_function(
        data=data, benchmark=benchmark,
        time_budget=time_budget,
        max_epochs=max_epochs,
        configuration=configuration
    )

    results["configs"] = results["configs"].get_dictionary()

    with open(args.output_file, "w") as fh:
        json.dump(results, fh)


if __name__ == "__main__":
    main()

