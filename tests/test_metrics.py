# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

import pytest

from sklearn_crfsuite import metrics


y1 = [["x", "z", "y"], ["x", "x"]]
y2 = [["y", "z", "y"], ["x", "y"]]


def test_flat_accuracy():
    score = metrics.flat_accuracy_score(y1, y2)
    assert score == 3 / 5


def test_flat_precision():
    score = metrics.flat_precision_score(y1, y2, average='micro')
    assert score == 3 / 5


def test_flat_recall():
    score = metrics.flat_recall_score(y1, y2, average='micro')
    assert score == 3 / 5


def test_flat_fscore():
    score = metrics.flat_f1_score(y1, y2, average='macro')
    assert score == 2 / 3
    assert metrics.flat_fbeta_score(y1, y2, beta=1, average='macro') == score


@pytest.mark.xfail(reason="see https://github.com/TeamHG-Memex/sklearn-crfsuite/issues/1")
def test_flat_f1_score_binary():
    s = [["x", "y"], ["x", "y"]]
    score = metrics.flat_f1_score(s, s, average='weighted')
    assert score == 1.0


def test_sequence_accuracy():
    assert metrics.sequence_accuracy_score(y1, y2) == 0
    assert metrics.sequence_accuracy_score([], []) == 0
    assert metrics.sequence_accuracy_score([[1,2], [3], [4]], [[1,2], [4], [4]]) == 2 / 3
    assert metrics.sequence_accuracy_score([[1,2], [3]], [[1,2], [3]]) == 1.0
