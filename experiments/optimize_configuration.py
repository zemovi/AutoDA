from keras.datasets import mnist, cifar10, cifar100
import hpbandster
import hpbandster.distributed.utils
from hpbandster.distributed.worker import Worker
from hpbandster.config_generators.kde_ei import KDEEI

import json
import sys
import os
import argparse
import Pyro4

import pickle

import logging
logging.basicConfig(level=logging.DEBUG)

from os.path import abspath, join as path_join
sys.path.insert(0, abspath(path_join(__file__, "..", "..")))

from autoda.networks.utils import get_data
from autoda.data_augmentation import ImageAugmentation
from autoda.networks.train import objective_function
from experiments.optimizers.hyperband import run_hpbandster
from experiments.optimizers.smac import run_smac


Pyro4.config.SERIALIZERS_ACCEPTED=set(["pickle"])
Pyro4.config.SERIALIZER="pickle"

def pickle_result(best_configuration, pickle_file):
    "pickles the result object of hyperband or BO-HB"

    with open(pickle_file, "wb") as fo:
        pickle.dump(best_configuration, fo)

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
        run_id=int(args.run_id),
    )

def benchmark_hpbandster(args):

    best_configuration = run_hpbandster(
        optimizer=args.optimizer,
        pipeline=args.pipeline,
        config_space=args.config_space,
        time_budget=int(args.time_budget),
        benchmark=args.benchmark,
        data=args.data,
        max_epochs=int(args.max_epochs),
        batch_size=int(args.batch_size)
    )


    pickle_result(best_configuration, args.pickle_file)
    # Config_id of the incumbent with smallest loss
    id_ = best_configuration.get_incumbent_id()
    run_info = best_configuration.get_runs_by_id(id_)[-1].__dict__
    trajectory = best_configuration.get_incumbent_trajectory()

    return to_json(
        output_file=args.output_file,
        best_configuration=(id_, run_info, trajectory),
        dataset=args.dataset,
        run_id=int(args.run_id),
    )



def main():
    # command line argument parser
    parser = argparse.ArgumentParser(description='Simple python script to run experiments on augmented data using random search')

    subparsers = parser.add_subparsers(dest="optimizer")

    smac_parser= subparsers.add_parser("SMAC", help="SMAC optimizer")
    smac_parser.set_defaults(func=benchmark_smac)

    hyperband_parser= subparsers.add_parser(
            "hyperband",
            help="Hyperband optimizer on steriods")

    hyperband_parser.set_defaults(func=benchmark_hpbandster)

    bohb_parser= subparsers.add_parser(
            "BOHB",
            help="Hyperband optimizer on steriods")

    bohb_parser.set_defaults(func=benchmark_hpbandster)

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
        default=500,
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
    # XXX: Remove augment if possible, augmentation is assumed if to optimize
    parser.add_argument(
        "--augment", action="store_true", help="If the data should be augmented, if flag not set defaults to false"
    )

    parser.add_argument(
        "--time-budget", default=7200, help="Maximum time budget to train a network",
        type=int, dest="time_budget"
    )
    parser.add_argument(
        "--dataset", default="cifar10", help="Dataset to train neural network on"
    )
    parser.add_argument(
        "--run-id", help="The id of single job", dest="run_id"
    )

    parser.add_argument(
    "--output-file", "-o", help="Output File",
    default=None, dest="output_file"
    )

    parser.add_argument(
    "--pickle-file", help="Output File",
    default=None, dest="pickle_file"
    )

    args = parser.parse_args()
    assert args.pipeline is not None
    assert args.run_id is not None
    assert args.dataset is not None
    assert args.optimizer is not None

    print(args.pipeline, args.dataset, args.run_id)

    optimizer_name = args.optimizer

    default_output_file = path_join(
        abspath("."), "AutoData",
        args.dataset,
        args.optimizer,
        args.benchmark,
        "best_config_{optimizer}_{dataset}_{run_id}.json".format(
            optimizer=args.optimizer, dataset=args.dataset, run_id=int(args.run_id)
        )
    )

    args.output_file = args.output_file or default_output_file

    default_pickle_file = path_join(
        abspath("."), "AutoData",
        args.dataset,
        args.optimizer,
        args.benchmark,
        "pickles",
        "{optimizer}_{dataset}_{run_id}.pickle".format(
            optimizer=args.optimizer, dataset=args.dataset, run_id=int(args.run_id)
            )
        )
    args.pickle_file = default_pickle_file

    assert args.pickle_file is not None
    assert args.output_file is not None

    configuration = None

    dataset = {"mnist": mnist, "cifar10": cifar10, "cifar100": cifar100}[args.dataset]


    args.config_space = ImageAugmentation.get_config_space()
    args.data = get_data(dataset, args.augment)

    args.func(args)


if __name__ == "__main__":
    main()
