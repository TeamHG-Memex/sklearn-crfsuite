================
sklearn-crfsuite
================

.. image:: https://img.shields.io/pypi/v/sklearn-crfsuite.svg
   :target: https://pypi.python.org/pypi/sklearn-crfsuite
   :alt: PyPI Version

.. image:: https://img.shields.io/travis/TeamHG-Memex/sklearn-crfsuite/master.svg
   :target: http://travis-ci.org/TeamHG-Memex/sklearn-crfsuite
   :alt: Build Status

.. image:: http://codecov.io/github/TeamHG-Memex/sklearn-crfsuite/coverage.svg?branch=master
   :target: http://codecov.io/github/TeamHG-Memex/sklearn-crfsuite?branch=master
   :alt: Code Coverage

.. image:: https://readthedocs.org/projects/sklearn-crfsuite/badge/?version=latest
   :target: http://sklearn-crfsuite.readthedocs.org/en/latest/?badge=latest
   :alt: Documentation

sklearn-crfsuite is a thin CRFsuite_ (python-crfsuite_) wrapper which provides
interface simlar to scikit-learn_. ``sklearn_crfsuite.CRF`` is a scikit-learn
compatible estimator: you can use e.g. scikit-learn model
selection utilities (cross-validation, hyperparameter optimization) with it,
or save/load CRF models using joblib_.

.. _CRFsuite: http://www.chokkan.org/software/crfsuite/
.. _python-crfsuite: https://github.com/scrapinghub/python-crfsuite
.. _scikit-learn: http://scikit-learn.org/
.. _joblib: https://github.com/joblib/joblib

License is MIT.

Documentation can be found `here <http://sklearn-crfsuite.readthedocs.org>`_.
