========
AutoDA
========

|Build Status|
|Docs_|
|Coverage_|
|Health_|

AutoDA is a Python framework for automated real-time data augmentation
for Deep Learning.

Features
========

** Data augmentation pipeline**, applying an augmentation pipeline with optimized parameters can be done simply by:

..  code-block::python
    augmented_data = augment(data)

Based on `keras <https://keras.io/>`_ that provides:
    * Allows for easy and fast prototyping (through user friendliness, modularity, and extensibility).
    * Supports both convolutional networks and recurrent networks, as well as combinations of the two.
    * Runs seamlessly on CPU and GPU.


.. |Build Status| image:: https://travis-ci.org/NMisgana/AutoDA.svg?branch_master
                  :target: https://travis-ci.org/NMisgana/AutoDA

.. |Docs_| image:: https://readthedocs.org/projects/AutoDA/badge/?version=latest
           :target: http://autoda.readthedocs.io/en/latest/
           :alt: Docs

.. |Coverage_| image:: https://coveralls.io/repos/github/NMisgana/AutoDA/badge.svg
               :target: https://coveralls.io/github/NMisgana/AutoDA
               :alt: Coverage

.. |Health_| image:: https://landscape.io/github/NMisgana/AutoDA/master/landscape.svg?style=flat
             :target: https://landscape.io/github/NMisgana/AutoDA/master
             :alt: Code Health


Install
=======

The quick way::

    pip3 install git+https://github.com/NMisgana/AutoDA


Documentation
=============
Documentation is available at http://autoda.readthedocs.io/en/latest/.
