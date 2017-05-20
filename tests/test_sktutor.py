#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_sktutor
----------------------------------

Tests for `sktutor` module.
"""

import pytest


#from sktutor import sktutor
import sktutor

@pytest.fixture
def response():
    """Sample pytest fixture.
    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    return 'asdf'


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument.
    """
    assert 'as' in response
