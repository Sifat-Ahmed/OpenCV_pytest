import math

import pytest

"""
def test_sqrt():
    num = 25
    assert math.sqrt(25) == 5

def test_square():
    num = 7
    assert num ** 2 == 44

def test_equal():
    assert 10 == 11

def test_greater():
    num = 100
    assert num > 100

def test_greater_equal():
    num = 100
    assert num >= 100

def test_less():
    num = 100
    assert num < 100
"""

@pytest.mark.great
def test_greater():
    num = 100
    assert num > 100

@pytest.mark.great
def test_greater_equal():
    num = 100
    assert num >= 100

@pytest.mark.other
def test_less():
    num = 100
    assert num < 100
