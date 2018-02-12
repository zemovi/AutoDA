#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-

import sys
import os
import argparse
import json
import numpy as np

from os.path import join as path_join, dirname, realpath, abspath

from robo.fmin import fabolas

PARENT_DIRECTORY = path_join(dirname(realpath(__file__)), "..", "..")
sys.path.insert(0, PARENT_DIRECTORY)

from autoda.data_augmentation import ImageAugmentation
from experiments.benchmarks.lenet_benchmark import lenet_function


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "run_id"
    )
    parser.add_argument(
        "dataset"
    )
    parser.add_argument(
        "num_iterations", type=int
    )
    parser.add_argument(
        "s_min"
    )
    parser.add_argument(
        "s_max"
    )
#    parser.add_argument(
#        "subsets"
#    )
#    parser.add_argument(
#        "seed"
#    )

    args = parser.parse_args()

    benchmark_function = {
        "LeNet": lenet_function,
    }[args.dataset]

    s_min, s_max = int(args.s_min), int(args.s_max)

    config_space = ImageAugmentation.get_config_space()

    hyperparameters = config_space.get_hyperparameters()

    lower = []
    upper = []

    for hyperparameter in hyperparameters:
        if hasattr(hyperparameter, "lower"):
            lower_bound = hyperparameter.lower
            upper_bound = hyperparameter.upper
        else:
            domain = hyperparameter.choices
            lower_bound, upper_bound = min(domain), max(domain)

        lower.append(lower_bound)
        upper.append(upper_bound)

    lower = np.array(lower)
    upper = np.array(upper)

    # Start Fabolas to optimize the objective function
    results = fabolas(
        lambda x, s: benchmark_function(x=x, s=int(s), config_space=config_space),
        lower=lower, upper=upper, s_min=s_min, s_max=s_max, num_iterations=50
    )

    x_best = results["x_opt"]
    print("best configuration", x_best)
    print(benchmark_function(x_best[:, :-1], s=x_best[:, None, -1]))

    path = path_join(abspath("."), "Workspace/MastersThesis/AutoDA/experiments/results/mnist/fabolas")
    with open(os.path.join(path, "fabolas_optimized_%d.json" % args.run_id), "w") as fh:
        json.dump(x_best, fh)


if __name__ == "__main__":
    main()
