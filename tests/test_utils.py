#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_preprocessing
----------------------------------

Tests for `preprocessing` module.
"""

import pytest
from sktutor.utils import (dict_factory, dict_default, bitwise_operator)
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
class TestBitwiseOperator(object):

    def test_or_binary_data(self, binary_data):
        # test bitwise_operator on binary data
        result = bitwise_operator(binary_data.iloc[:, 0:2], 'or')
        expected = pd.Series([1, 1, 0, 1])
        tm.assert_series_equal(result, expected, check_dtype=False)

    def test_or_boolean_data(self, boolean_data):
        # test bitwise_operator on boolean data
        result = bitwise_operator(boolean_data.iloc[:, 0:2], 'or')
        expected = pd.Series([True, True, False, True])
        tm.assert_series_equal(result, expected, check_dtype=False)

    def test_or_binary_series(self, binary_series):
        # test bitwise_operator on boolean data
        df = pd.DataFrame(binary_series)
        result = bitwise_operator(df, 'or')
        print(df)
        print(result)
        expected = pd.Series([1, 1, 0, 0])
        tm.assert_series_equal(result, expected, check_names=False)

    def test_and_binary_data(self, binary_data):
        # test bitwise_operator on binary data
        result = bitwise_operator(binary_data.iloc[:, 0:2], 'and')
        expected = pd.Series([1, 0, 0, 0])
        tm.assert_series_equal(result, expected, check_dtype=False)

    def test_and_boolean_data(self, boolean_data):
        # test bitwise_operator on boolean data
        result = bitwise_operator(boolean_data.iloc[:, 0:2], 'and')
        expected = pd.Series([True, False, False, False])
        tm.assert_series_equal(result, expected, check_dtype=False)

    def test_and_binary_series(self, binary_series):
        # test bitwise_operator on boolean data
        df = pd.DataFrame(binary_series)
        result = bitwise_operator(df, 'or')
        print(df)
        print(result)
        expected = pd.Series([1, 1, 0, 0])
        tm.assert_series_equal(result, expected, check_names=False)
