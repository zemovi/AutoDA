import hpbandster
import hpbandster.distributed.utils
from hpbandster.distributed.worker import Worker

import pickle
import json
import sys
import os
from os.path import abspath, join as path_join
sys.path.insert(0, abspath(path_join(__file__, "..", "..")))

from autoda.data_augmentation import ImageAugmentation
from autoda.networks.train import objective_function
from keras.datasets import cifar10

import logging
logging.basicConfig(level=logging.DEBUG)

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

            For dramatization, the function sleeps for one second, which emphasizes
            the speed ups achievable with parallel workers.
            """

            results = self.function(
                benchmark="AlexNet", configuration=config, dataset=self.dataset, max_epochs=40, batch_size=512, time_budget=budget
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

# this needs to be unique for concurent runs, i.e. when multiple
# instances run at the same time, they have to have different ids
# For this all_in_one example, it doesn't reall matter, as the
# nameserver is unique to this run

# instantiating Hyperband with some minimal configuration
HB = hpbandster.HB_master.HpBandSter(
    config_generator=CG,
    run_id='0',
    eta=2,
    min_budget=50,
    max_budget=1800,      # HB parameters
    nameserver=nameserver,
    ns_port=ns_port,
    job_queue_sizes=(0, 1)
)
# runs one iteration if at least one worker is available, first parameter
# is number of successive halving
res = HB.run(5, min_n_workers=1)
# shutdown the worker and the dispatcher
HB.shutdown(shutdown_workers=True)


path = path_join(abspath("."), "AutoData/hyperband")

with open(os.path.join(path, 'hyperband.pickle'), 'wb') as f:
    # Pickle the 'res' object from hyperband to enable us to extract useful information
    pickle.dump(res, f)


print(res.get_incumbent_trajectory())
best_config = res.get_incumbent_trajectory()


with open(os.path.join(path, "hyperband_result.json"), "w") as fh:
        json.dump(best_config, fh)
