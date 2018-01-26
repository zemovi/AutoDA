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


from keras.datasets import mnist, cifar10
sys.path.insert(0, abspath(path_join(__file__, "..", "..")))

from autoda.data_augmentation import ImageAugmentation
from autoda.networks.train import objective_function
from autoda.networks.utils import get_data


def main():

    parser = argparse.ArgumentParser(description='Simple python script to benchmark data augmentation configurations.')

    parser.add_argument(
        "--benchmark",
        help="Neural network to be trained with augmented data"
    )

    parser.add_argument(
        "--max_epochs", default=500, type=int,
        help="Maximum number of epochs to train network")

    parser.add_argument(
        "--batch_size", default=512, type=int,
        help="Size of a mini batch"
    )

    parser.add_argument(
        "--time_budget", default=1800, type=int,
        help="Maximum time budget to train a network"
    )

    parser.add_argument(
        "--dataset",
        help="Dataset to train neural network on"
    )

    parser.add_argument(
        "--run_id",
        help="The id of single job"
    )

    parser.add_argument(
        "--configuration_file",
        help="Path to JSON file containing a configuration dictionary for "
             "our data augmentation. "
             "Defaults to `None`, which uses no data augmentation.",
        default=None, dest="configuration_file"
    )

    parser.add_argument(
        "--output_file", "-o", help="Output File",
        default=None, dest="output_file"
    )

    args = parser.parse_args()

    default_outputfile = path_join(
        abspath("."),
        "AutoData",
        args.dataset,
        "best_config_v1/best_config_{dataset}_{run_id}.json".format(
            dataset=args.dataset, run_id=int(args.run_id)
        )
    )

    args.output_file = args.output_file or default_outputfile

    configuration = None

    if args.configuration_file:
        with open(args.configuration_file, "r") as configuration_file:
            configuration = json.load(configuration_file)["best_run_info"]["configs"] # XXX: rewrite json format to fit optimized results from hyperband, smac or defualt
            print("configuration_file", configuration)

    augment = args.configuration_file is not None

    benchmark = args.benchmark

    dataset = {"mnist": mnist, "cifar10": cifar10}[args.dataset]

    max_epochs, batch_size, time_budget, augment = int(args.max_epochs), int(args.batch_size), int(args.time_budget), augment

    data = get_data(dataset, augment)

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

