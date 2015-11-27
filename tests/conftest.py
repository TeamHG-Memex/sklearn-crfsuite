# -*- coding: utf-8 -*-
from __future__ import absolute_import
import pytest


@pytest.fixture()
def xseq():
    return [
        {'walk': 1, 'shop': 0.5},
        {'walk': 1},
        {'walk': 1, 'clean': 0.5},
        {u'shop': 0.5, u'clean': 0.5},
        {'walk': 0.5, 'clean': 1},
        {'clean': 1, u'shop': 0.1},
        {'walk': 1, 'shop': 0.5},
        {},
        {'clean': 1},
        {u'солнце': u'не светит'.encode('utf8'), 'clean': 1},
    ]

@pytest.fixture
def yseq():
    return ['sunny', 'sunny', u'sunny', 'rainy', 'rainy', 'rainy',
            'sunny', 'sunny', 'rainy', 'rainy']
