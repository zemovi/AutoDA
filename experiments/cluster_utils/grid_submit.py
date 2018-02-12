#!/usr/bin/python3
# -*- encoding: utf-8 -*-
import argparse
from itertools import product
from os.path import join as path_join
from subprocess import check_call

check_call = lambda t: print(" ".join(t))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        help="(Comma-seperated) string of datasets to use. Defaults to `cifar10`.",
        default="cifar10"
    )

    parser.add_argument(
        "--benchmarks",
        help="(Comma-seperated) string of models to use. Defaults to `AlexNet`.",
        default="AlexNet"
    )

    parser.add_argument(
        "--no-runs", help="Number of runs to perform.", type=int, dest="no_runs",
        default=1
    )

    parser.add_argument(
        "submit_script", help="Submit script for this job."
    )

    parser.add_argument(
        "output_directory", help="Directory for output files."
    )

    subparsers = parser.add_subparsers()
    optimize_parser = subparsers.add_parser("optimize")
    optimize_parser.add_argument(
        "optimizers",
        help="(Comma-seperated) string of optimizers to use. Defaults to `hyperband`.",
    )

    args = parser.parse_args()

    datasets = args.datasets.split(",") * args.no_runs
    benchmarks = args.benchmarks.split(",")
    try:
        optimizers = args.optimizers.split(",")
    except AttributeError:
        for run_id, (dataset, benchmark) in enumerate(product(datasets, benchmarks)):
            check_call((
                "qsub", "-v",
                "dataset={},run_id={},benchmark={}".format(dataset, run_id, benchmark),
                "-q", "gpu", "-l", "nodes=1:ppn=1:gpus=1",
                "-o", path_join(args.output_directory, "of_run_{run_id}".format(run_id=run_id)),
                "-e", path_join(args.output_directory, "ef_run_{run_id}".format(run_id=run_id)),
                args.submit_script
            ))
    else:
        for run_id, (dataset, optimizer, benchmark) in enumerate(product(datasets, optimizers, benchmarks)):
            check_call((
                "qsub", "-v",
                "dataset={},opt={},benchmark={},run_id={}".format(dataset, optimizer, benchmark, run_id),
                "-q", "gpu", "-l", "nodes=1:ppn=1:gpus=1",
                "-o", path_join(args.output_directory, "of_run_{run_id}".format(run_id=run_id)),
                "-e", path_join(args.output_directory, "ef_run_{run_id}".format(run_id=run_id)),
                args.submit_script
            ))


if __name__ == "__main__":
    main()
