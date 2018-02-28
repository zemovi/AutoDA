
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
sys.path.insert(0, abspath(path_join(__file__, "..", "..", "..")))

from autoda.networks.utils import get_data
from autoda.data_augmentation import ImageAugmentation
from autoda.networks.train import objective_function

# this run hyperband sequentially

class ImageAugmentationWorker(Worker):
    def __init__(self, experiment, pipeline,
                 max_epochs, batch_size, *args, **kwargs):
        self.benchmark, self.data = experiment
        self.pipeline=pipeline
        self.max_epochs, self.batch_size = max_epochs, batch_size
        super().__init__(*args, **kwargs)

    def compute(self, config, budget, *args, **kwargs):

        if self.pipeline == "standard":
            # XXX: move this to its own function
            # XXX: check out if config space can be called with default values(get_config_space().get_default_value)
            print("USING STANDARD AUGMENTATION")
            config['scale_probability'] = 0.
            config['rotation_probability'] = 0.
            config['rotation_lower'] = 0
            config['rotation_upper'] = 0
            config['scale_probability'] = 0.
            config['scale_upper'] = 1.
            config['scale_lower'] = 1.
            config['vertical_flip'] = 0.
            config['coarse_dropout_probability'] = 0.
            config['coarse_dropout_lower'] = 0.
            config['coarse_dropout_upper'] = 0.1
            config['coarse_dropout_size_percent_upper'] = 0.1
            config['coarse_dropout_size_percent_lower'] = 0.01

        results = objective_function(
            benchmark=self.benchmark, configuration=config,
            data=self.data, max_epochs=self.max_epochs,
            batch_size=self.batch_size, time_budget=budget
        )

        validation_error = results["validation_error"]

        return({
            'loss': validation_error,   # this is the a mandatory field to run hyperband
            'info': results  # can be used for any user-defined information - also mandatory
        })


def run_hpbandster(model_based, pipeline, config_space, time_budget, benchmark, data, max_epochs, batch_size):
    # starts a local nameserver
    nameserver, ns_port = hpbandster.distributed.utils.start_local_nameserver()

    # starting the worker in a separate thread
    w = ImageAugmentationWorker(
        nameserver=nameserver, ns_port=ns_port,
        experiment=(benchmark, data),
        pipeline=pipeline,
        max_epochs=max_epochs,
        batch_size=batch_size,
    )

    w.run(background=True)


    # simple config space here: just one float between 0 and 1

    if model_based:
        print("Using Model Based Hyperband")
        CG = KDEEI(config_space, mode="sampling", num_samples=64)  # model-based hyperband
    else:
        print("Using Hyperband")
        CG = hpbandster.config_generators.RandomSampling(config_space)  # hyperband on steriods


# XXX: change min_budget to 500
    # instantiating Hyperband with some minimal configuration
    HB = hpbandster.HB_master.HpBandSter(
        config_generator=CG,
        run_id='0',
        eta=2,
        min_budget=500,
        max_budget=time_budget,
        nameserver=nameserver,
        ns_port=ns_port,
        job_queue_sizes=(0, 1)
    )
    # runs one iteration if at least one worker is available, first parameter
    # is number of successive halving
    res = HB.run(5, min_n_workers=1)
    # shutdown the worker and the dispatcher
    HB.shutdown(shutdown_workers=True)

    return res

