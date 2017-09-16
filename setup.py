#!/usr/bin/python3
# -*- encoding: utf-8 -*-
from os.path import realpath, dirname, join
from setuptools import setup, find_packages


NAME = "autoda"
DESCRIPTION = "Auto-DataAugmentation"
LONG_DESCRIPTION = """AutoDA is a Python framework for automated real-time data augmentation"""
MAINTAINER = "Misgana Negassi"
MAINTAINER_EMAIL = "misganazemo@gmail.com"
URL = "https://github.com/NMisgana/AutoDA"
# LICENSE = ""
VERSION = "0.0.1"


classifiers = [
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Development Status :: 3 - Alpha',
    'Natural Language :: English',
    'Environment :: Console',
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    'Operating System :: POSIX :: Linux',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Scientific/Engineering',
    'Topic :: Software Development',
]

PROJECT_ROOT = dirname(realpath(__file__))
REQUIREMENTS_FILE = join(PROJECT_ROOT, 'requirements.txt')

with open(REQUIREMENTS_FILE, "r") as f:
    INSTALL_REQUIREMENTS = f.read().splitlines()


SETUP_REQUIREMENTS = ["pytest-runner"]
TEST_REQUIREMENTS = ["pytest", "pytest-cov", "hypothesis"]


if __name__ == "__main__":
        setup(name=NAME,
              version=VERSION,
              maintainer=MAINTAINER,
              maintainer_email=MAINTAINER_EMAIL,
              description=DESCRIPTION,
              url=URL,
              long_description=LONG_DESCRIPTION,
              packages=find_packages(),
              package_data={'docs': ['*']},
              include_package_data=True,
              classifiers=classifiers,
              install_requires=INSTALL_REQUIREMENTS,
              setup_requires=SETUP_REQUIREMENTS,
              tests_require=TEST_REQUIREMENTS,
              )
