# -*- coding: utf-8 -*-
from __future__ import absolute_import, division
from functools import wraps

from sklearn import metrics as _metrics

from sklearn_crfsuite.utils import flatten


def _flattens_y(func):
    @wraps(func)
    def wrapper(y_true, y_pred, *args, **kwargs):
        y_true_flat = flatten(y_true)
        y_pred_flat = flatten(y_pred)
        return func(y_true_flat, y_pred_flat, *args, **kwargs)
    return wrapper


@_flattens_y
def flat_accuracy_score(y_true, y_pred):
    """
    Return accuracy score for sequence items.
    """
    return _metrics.accuracy_score(y_true, y_pred)


@_flattens_y
def flat_f1_score(y_true, y_pred, **kwargs):
    """
    Return F1 score for sequence items.
    """
    return _metrics.f1_score(y_true, y_pred, **kwargs)


@_flattens_y
def flat_fbeta_score(y_true, y_pred, beta, **kwargs):
    """
    Return F-beta score for sequence items.
    """
    return _metrics.fbeta_score(y_true, y_pred, beta, **kwargs)


@_flattens_y
def flat_classification_report(y_true, y_pred, labels=None, **kwargs):
    """
    Return classification report for sequence items.
    """
    return _metrics.classification_report(y_true, y_pred, labels, **kwargs)


def sequence_accuracy_score(y_true, y_pred):
    """
    Return sequence accuracy score. Match is counted only when two sequences
    are equal.
    """
    total = len(y_true)
    if not total:
        return 0

    matches = sum(1 for yseq_true, yseq_pred in zip(y_true, y_pred)
                  if yseq_true == yseq_pred)

    return matches / total

