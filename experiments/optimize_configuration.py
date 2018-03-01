from keras.datasets import mnist, cifar10, cifar100
import hpbandster
import hpbandster.distributed.utils
from hpbandster.distributed.worker import Worker
from hpbandster.config_generators.kde_ei import KDEEI

import json
import sys
import os
import argparse

import logging
logging.basicConfig(level=logging.DEBUG)

from os.path import abspath, join as path_join
sys.path.insert(0, abspath(path_join(__file__, "..", "..")))

from autoda.networks.utils import get_data
from autoda.data_augmentation import ImageAugmentation
from autoda.networks.train import objective_function
from experiments.optimizers.hyperband import run_hpbandster
from experiments.optimizers.smac import run_smac


def to_json(output_file, best_configuration, dataset, run_id):
        id_, run_info, trajectory = best_configuration
        json_data = {
            "best_config_id": id_,
            "best_run_info": run_info,
            "best_config_trajectory": trajectory
        }

        with open(output_file, "w") as fh:
            json.dump(json_data, fh)

def benchmark_smac(args):

    best_configuration = run_smac(
        config_space=args.config_space,
        time_budget=int(args.time_budget),
        benchmark=args.benchmark,
        data=args.data,
        max_epochs=int(args.max_epochs),
        batch_size=int(args.batch_size)
            )

    return to_json(
        output_file=args.output_file,
        best_configuration=(id_, run_info, trajectory),
        dataset=args.dataset,
        run_id=args.run_id,
    )

def benchmark_hpbandster(args):

    best_configuration = run_hpbandster(
        model_based=args.model_based,
        pipeline=args.pipeline,
        config_space=args.config_space,
        time_budget=int(args.time_budget),
        benchmark=args.benchmark,
        data=args.data,
        max_epochs=int(args.max_epochs),
        batch_size=int(args.batch_size)
    )

    # Config_id of the incumbent with smallest loss
    id_ = best_configuration.get_incumbent_id()
    run_info = best_configuration.get_runs_by_id(id_)[-1]
    trajectory = best_configuration.get_incumbent_trajectory()

    return to_json(
        output_file=args.output_file,
        best_configuration=(id_, run_info, trajectory),
        dataset=args.dataset,
        run_id=args.run_id,
    )



def main():
    # command line argument parser
    parser = argparse.ArgumentParser(description='Simple python script to run experiments on augmented data using random search')

    subparsers = parser.add_subparsers(dest="optimizer")

    smac_parser= subparsers.add_parser("SMAC", help="SMAC optimizer")
    smac_parser.set_defaults(func=benchmark_smac)

    hpbandster_parser= subparsers.add_parser(
            "hyperband",
            help="Hyperband optimizer on steriods")

    hpbandster_parser.add_argument(
        "--model-based",
        action="store_true",
        dest="model_based",
        help="Model-based hyperband"
    )

    hpbandster_parser.set_defaults(func=benchmark_hpbandster)

    parser.add_argument(
        "--benchmark",
        help="Neural network to be trained with augmented data"
    )

    parser.add_argument(
        "--pipeline",
        dest="pipeline",
        help="Data augmentation pipeline to use, choice:{standard, pipeline1, pipeline2}"
    )

    parser.add_argument(
        "--max-epochs",
        default=500,
        dest="max_epochs",
        type=int,
        help="Maximum number of epochs to train network"
    )
    parser.add_argument(
        "--batch-size", default=512,
        dest="batch_size",
        type=int,
        help="Size of a mini batch",
    )
    # XXX: Remove augment if possible, augmentation is assumed if to optimize
    parser.add_argument(
        "--augment", action="store_true", help="If the data should be augmented, if flag not set defaults to false"
    )

    parser.add_argument(
        "--time-budget", default=7200, help="Maximum time budget to train a network",
        type=int, dest="time_budget"
    )
    parser.add_argument(
        "--dataset", help="Dataset to train neural network on"
    )
    parser.add_argument(
        "--run-id", help="The id of single job", dest="run_id"
    )

    parser.add_argument(
    "--output-file", "-o", help="Output File",
    default=None, dest="output_file"
    )

    args = parser.parse_args()

    optimizer_name = args.optimizer

    if optimizer_name == "hyperband":
        if args.model_based:
            optimizer_name = "BOHB"

    default_output_file = path_join(
        abspath("."), "AutoData",
        args.dataset,
        "hyperband/best_config_{optimizer}_{dataset}_{run_id}.json".format(
            optimizer=args.optimizer, dataset=args.dataset, run_id=int(args.run_id)
        )
    )

    args.output_file = args.output_file or default_output_file


    configuration = None

    dataset = {"mnist": mnist, "cifar10": cifar10, "cifar100": cifar100}[args.dataset]

    # max_epochs, batch_size, time_budget, optimizer, augment, run_id = int(args.max_epochs), int(args.batch_size), int(args.time_budget), args.optimizer, args.augment, args.run_id

    args.config_space = ImageAugmentation.get_config_space()
    args.data = get_data(dataset, args.augment)

    args.func(args)


if __name__ == "__main__":
    main()
