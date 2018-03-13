import numpy as np
import keras
import json
import sys
import os

from os.path import dirname, abspath, realpath, join as path_join

from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

from sklearn.model_selection import train_test_split

PARENT_DIRECTORY = path_join(dirname(realpath(__file__)), "..", "..")
sys.path.insert(0, PARENT_DIRECTORY)

from autoda.data_augmentation import ImageAugmentation

from autoda.networks.train import objective_function
# XXX: Run test script if works and FIX problems

def run_smac(config_space, time_budget, benchmark, data, max_epochs, batch_size):

    # Scenario object
    scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
                         "runcount-limit": 100,  # maximum function evaluations
                         "cs": config_space,     # configuration space
                         "deterministic": "true"
                         })

    # Optimize, using a SMAC-object
    print("Optimizing with SMAC!")
    smac = SMAC(scenario=scenario, rng=np.random.RandomState(42),
            tae_runner=objective_function)

    incumbent = smac.optimize()

    incumbent_value = train_with_config(incumbent)

    return objective_function(
        benchmark=benchmark, configuration=incumbent,
        data=data, max_epochs=max_epochs,
        batch_size=batch_size, time_budget=time_budget
    )

