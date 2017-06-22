================
sklearn-crfsuite
================

.. image:: https://img.shields.io/pypi/v/sklearn-crfsuite.svg
   :target: https://pypi.python.org/pypi/sklearn-crfsuite
   :alt: PyPI Version

.. image:: https://img.shields.io/travis/TeamHG-Memex/sklearn-crfsuite/master.svg
   :target: https://travis-ci.org/TeamHG-Memex/sklearn-crfsuite
   :alt: Build Status

.. image:: https://codecov.io/github/TeamHG-Memex/sklearn-crfsuite/coverage.svg?branch=master
   :target: https://codecov.io/github/TeamHG-Memex/sklearn-crfsuite?branch=master
   :alt: Code Coverage

.. image:: https://readthedocs.org/projects/sklearn-crfsuite/badge/?version=latest
   :target: https://sklearn-crfsuite.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation

sklearn-crfsuite is thin a CRFsuite_ (python-crfsuite_) wrapper which provides
scikit-learn_-compatible :class:`sklearn_crfsuite.CRF` estimator:
you can use e.g. scikit-learn model selection utilities
(cross-validation, hyperparameter optimization) with it, or save/load CRF
models using joblib_.

.. _CRFsuite: http://www.chokkan.org/software/crfsuite/
.. _python-crfsuite: https://github.com/scrapinghub/python-crfsuite
.. _scikit-learn: http://scikit-learn.org/
.. _joblib: https://github.com/joblib/joblib

License is MIT.

Contents
========

.. toctree::
   :maxdepth: 2

   install
   tutorial
   api
   contributing
   changes
