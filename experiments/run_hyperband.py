import hpbandster
import hpbandster.distributed.utils
from hpbandster.distributed.worker import Worker

import json
import sys
import os

import logging
logging.basicConfig(level=logging.DEBUG)

from os.path import abspath, join as path_join
sys.path.insert(0, abspath(path_join(__file__, "..", "..")))

from autoda.data_augmentation import ImageAugmentation
from autoda.networks.train import objective_function
from keras.datasets import cifar10


# this run hyperband sequentially


class ImageAugmentationWorker(Worker):
        def __init__(self, benchmark=(objective_function, cifar10), *args, **kwargs):
            self.function, self.dataset = benchmark
            super().__init__(*args, **kwargs)

        def compute(self, config, budget, *args, **kwargs):
            """
            Simple example for a compute function

            The loss is just a the config + some noise (that decreases with the budget)
            There is a 10 percent failure probability for any run, just to demonstrate
            the robustness of Hyperband agains these kinds of failures.
            """

            results = self.function(
                benchmark="AlexNet", configuration=config, dataset=self.dataset, max_epochs=100, batch_size=512, time_budget=budget
            )

            validation_error = results["validation_error"]
            # loss = results

            return({
                'loss': validation_error,   # this is the a mandatory field to run hyperband
                'info': results  # can be used for any user-defined information - also mandatory
            })


# starts a local nameserve
nameserver, ns_port = hpbandster.distributed.utils.start_local_nameserver()


# import the definition of the worker (could be in here as well, but is imported to reduce code duplication)

# starting the worker in a separate thread
w = ImageAugmentationWorker(nameserver=nameserver, ns_port=ns_port)
w.run(background=True)


# simple config space here: just one float between 0 and 1
config_space = ImageAugmentation.get_config_space()
CG = hpbandster.config_generators.RandomSampling(config_space)


# instantiating Hyperband with some minimal configuration
HB = hpbandster.HB_master.HpBandSter(
    config_generator=CG,
    run_id='0',
    eta=2,
    min_budget=50,
    max_budget=3600,
    nameserver=nameserver,
    ns_port=ns_port,
    job_queue_sizes=(0, 1)
)
# runs one iteration if at least one worker is available, first parameter
# is number of successive halving
res = HB.run(5, min_n_workers=1)
# shutdown the worker and the dispatcher
HB.shutdown(shutdown_workers=True)


# Save results
path = path_join(abspath("."), "AutoData/hyperband")

# Get important information about best configuration from HB result object
best_config_id = res.get_incumbent_id()  # Config_id of the incumbent with smallest loss
best_run = res.get_runs_by_id(best_config_id)[-1]
best_config_trajectory = res.get_incumbent_trajectory()


json_data = {
    "best_config_id": best_config_id,
    "best_run": best_run,
    "best_config_trajectory": best_config_trajectory
}

with open(os.path.join(path, "hyperband_result.json"), "w") as fh:
        json.dump(json_data, fh)
