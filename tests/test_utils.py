#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_preprocessing
----------------------------------

Tests for `preprocessing` module.
"""

import pytest
from sktutor.utils import (dict_factory, dict_default, bitwise_or, bitwise_and)
import pandas as pd
import pandas.util.testing as tm


# @pytest.mark.usefixtures("missing_data2")
class TestDictFactory(object):

    def test_dict_factory(self, missing_data):
        # test that a dict_factory dict has new default value for missing keys
        new_dict = dict_factory('new_dict', 'default')
        d = {'a': 1, 'b': 2}
        d2 = new_dict(d)
        expected = 'default'
        assert d2['c'] == expected


class TestDictDefault(object):

    def test_dict_default(self, missing_data):
        # test that a dict_factory dict has new default value for missing keys
        d = {'a': 1, 'b': 2}
        d2 = dict_default(d)
        expected = 'c'
        assert d2['c'] == expected


@pytest.mark.usefixtures("binary_data")
@pytest.mark.usefixtures("boolean_data")
@pytest.mark.usefixtures("binary_series")
class TestBitwiseOr(object):

    def test_binary_data(self, binary_data):
        # test bitwise_or on binary data
        result = bitwise_or(binary_data.iloc[:, 0:2])
        expected = pd.Series([1, 1, 0, 1])
        tm.assert_series_equal(result, expected, check_dtype=False)

    def test_boolean_data(self, boolean_data):
        # test bitwise_or on boolean data
        result = bitwise_or(boolean_data.iloc[:, 0:2])
        expected = pd.Series([True, True, False, True])
        tm.assert_series_equal(result, expected, check_dtype=False)

    def test_binary_series(self, binary_series):
        # test bitwise_or on boolean data
        df = pd.DataFrame(binary_series)
        result = bitwise_or(df)
        print(df)
        print(result)
        expected = pd.Series([1, 1, 0, 0])
        tm.assert_series_equal(result, expected, check_names=False)


@pytest.mark.usefixtures("binary_data")
@pytest.mark.usefixtures("boolean_data")
@pytest.mark.usefixtures("binary_series")
class TestBitwiseAnd(object):

    def test_binary_data(self, binary_data):
        # test bitwise_and on binary data
        result = bitwise_and(binary_data.iloc[:, 0:2])
        expected = pd.Series([1, 0, 0, 0])
        tm.assert_series_equal(result, expected, check_dtype=False)

    def test_boolean_data(self, boolean_data):
        # test bitwise_and on boolean data
        result = bitwise_and(boolean_data.iloc[:, 0:2])
        expected = pd.Series([True, False, False, False])
        tm.assert_series_equal(result, expected, check_dtype=False)

    def test_binary_series(self, binary_series):
        # test bitwise_and on boolean data
        df = pd.DataFrame(binary_series)
        result = bitwise_and(df)
        print(df)
        print(result)
        expected = pd.Series([1, 1, 0, 0])
        tm.assert_series_equal(result, expected, check_names=False)
