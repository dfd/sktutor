#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_preprocessing
----------------------------------

Tests for `preprocessing` module.
"""

import pytest
from sktutor.utils import (dict_factory, dict_default, bitwise_operator,
                           bitwise_xor, bitwise_not)
import pandas as pd
import pandas.testing as tm


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
        result = bitwise_operator(df, 'and')
        print(df)
        print(result)
        expected = pd.Series([1, 1, 0, 0])
        tm.assert_series_equal(result, expected, check_names=False)


@pytest.mark.usefixtures("binary_data")
@pytest.mark.usefixtures("boolean_data")
@pytest.mark.usefixtures("binary_series")
class TestBitwiseXor(object):

    def test_binary_data(self, binary_data):
        # test bitwise_xor binary data
        result = bitwise_xor(binary_data.iloc[:, 0:2])
        expected = pd.Series([0, 1, 0, 1])
        tm.assert_series_equal(result, expected, check_dtype=False)

    def test_boolean_data(self, boolean_data):
        # test bitwise_xor on boolean data
        result = bitwise_xor(boolean_data.iloc[:, 0:2])
        expected = pd.Series([False, True, False, True])
        tm.assert_series_equal(result, expected, check_dtype=False)

    def test_too_many_columns_value_error(self, binary_data):
        # Test throwing error for more than 2 columns
        with pytest.raises(ValueError):
            bitwise_xor(binary_data)

    def test_too_few_columns_value_error(self, binary_series):
        # Test throwing error for just 1 column
        with pytest.raises(ValueError):
            bitwise_xor(pd.DataFrame(binary_series))


@pytest.mark.usefixtures("binary_data")
@pytest.mark.usefixtures("boolean_data")
@pytest.mark.usefixtures("binary_series")
class TestBitwiseNot(object):

    def test_binary_data(self, binary_data):
        # test bitwise_not on binary data
        result = bitwise_not(binary_data)
        exp_dict = {'a': [False, False, True, True],
                    'b': [False, True, True, False],
                    'c': [True, False, False, True],
                    'd': [False, True, False, True],
                    'e': [True, False, True, False]
                    }
        expected = pd.DataFrame(exp_dict)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_boolean_data(self, boolean_data):
        # test bitwise_not on boolean data
        result = bitwise_not(boolean_data)
        exp_dict = {'a': [False, False, True, True],
                    'b': [False, True, True, False],
                    'c': [True, False, False, True],
                    'd': [False, True, False, True],
                    'e': [True, False, True, False]
                    }
        expected = pd.DataFrame(exp_dict)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_binary_series(self, binary_series):
        # test bitwise_xor on binary series
        result = bitwise_not(binary_series)
        expected = pd.Series([False, False, True, True])
        tm.assert_series_equal(result, expected, check_dtype=False)
