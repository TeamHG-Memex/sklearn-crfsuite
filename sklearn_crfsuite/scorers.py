# -*- coding: utf-8 -*-
"""
Scorer functions to use with scikit-learn ``cross_val_score``,
``GridSearchCV``, ``RandomizedSearchCV`` and other similar classes.
"""
from sklearn.metrics import make_scorer

from sklearn_crfsuite import metrics


flat_accuracy = make_scorer(metrics.flat_accuracy_score)
sequence_accuracy = make_scorer(metrics.sequence_accuracy_score)
