
from keras.datasets import mnist, cifar10
import hpbandster
import hpbandster.distributed.utils
from hpbandster.distributed.worker import Worker

import json
import sys
import os
import argparse
import pickle

import logging
logging.basicConfig(level=logging.DEBUG)

from os.path import abspath, join as path_join
sys.path.insert(0, abspath(path_join(__file__, "..", "..")))

from autoda.networks.utils import get_data
from autoda.data_augmentation import ImageAugmentation
from autoda.networks.train import objective_function

# command line arguement parser
parser = argparse.ArgumentParser(description='Simple python script to run experiments on augmented data using random search')

parser.add_argument(
    "--benchmark", help="Neural network to be trained with augmented data"
)
parser.add_argument(
    "--max_epochs", default=200, help="Maximum number of epochs to train network", type=int
)
parser.add_argument(
    "--batch_size", default=512, help="Size of a mini batch", type=int
)
parser.add_argument(
    "--augment", action="store_true", help="If the data should be augmented, if flag not set defaults to false"
)
parser.add_argument(
    "--dataset", help="Dataset to train neural network on"
)
parser.add_argument(
    "--time_budget", default=1800, help="Maximum time budget to train a network", type=int
)
parser.add_argument(
    "--run_id", help="The id of single job"
)

args = parser.parse_args()

benchmark = args.benchmark

dataset = {
    "mnist": mnist,
    "cifar10": cifar10
}[args.dataset]

max_epochs, batch_size, time_budget, augment = int(args.max_epochs), int(args.batch_size), int(args.time_budget), args.augment

data = get_data(dataset, augment)


# this run hyperband sequentially

class ImageAugmentationWorker(Worker):
	def __init__(self, benchmark=(objective_function, data), *args, **kwargs):
		self.function, self.data = benchmark
		super().__init__(*args, **kwargs)

	def compute(self, config, budget, *args, **kwargs):
		"""
		Simple example for a compute function

		The loss is just a the config + some noise (that decreases with the budget)
		There is a 10 percent failure probability for any run, just to demonstrate
		the robustness of Hyperband agains these kinds of failures.
		"""
		config['scale_probability'] = 0.
		config['rotation_probability'] = 0.
		config['vertical_flip'] = 0.
		config['coarse_dropout_probability'] = 0.
		config['elastic_transform_probability'] = 0.
		config['shear_probability'] = 0.
		config['gaussian_noise_probability'] = 0.

		print("DEFAULT_CONFIG", config)

		results = self.function(
			benchmark=benchmark, configuration=config, data=data, max_epochs=max_epochs, batch_size=batch_size, time_budget=budget
		)
		validation_error = results["validation_error"]

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

# Save results
path = path_join(abspath("."), "AutoData/{}/hyperband".format(args.dataset))

pickle.dump(res, open("hyperband.pkl", "wb"))


# Get important information about best configuration from HB result object
best_config_id = res.get_incumbent_id()  # Config_id of the incumbent with smallest loss
best_run = res.get_runs_by_id(best_config_id)[-1]
best_config_trajectory = res.get_incumbent_trajectory()


json_data = {
    "best_config_id": best_config_id,
    "best_run_info": best_run.info,
    "best_config_trajectory": best_config_trajectory
}

with open(os.path.join(path, "hyperband_optimized_default_augment_result_{}.json".format(time_budget)), "w") as fh:
	json.dump(json_data, fh)
