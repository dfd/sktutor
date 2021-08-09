#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from sktutor.preprocessing import (GroupByImputer, MissingValueFiller,
                                   ColumnExtractor, ColumnDropper)
from sktutor.pipeline import (FeatureUnion, make_union)
from sklearn.pipeline import make_pipeline
import pandas as pd
import pandas.testing as tm


@pytest.mark.usefixtures("missing_data")
class TestFeatureUnion(object):

    def test_feature_union(self, missing_data):
        # Test FeatureUnion
        CONTINUOUS_FIELDS = missing_data.select_dtypes(
            ['int64', 'float64']).columns.tolist()
        FACTOR_FIELDS = missing_data.select_dtypes(['object']).columns
        CONTINUOUS_FIELDS.append('b')
        fu = FeatureUnion(
            [('Continuous Pipeline', make_pipeline(
                ColumnExtractor(CONTINUOUS_FIELDS),
                GroupByImputer('median', 'b'),
                ColumnDropper('b'),
                GroupByImputer('median')
            )),
             ('Factor Pipeline', make_pipeline(
                ColumnExtractor(FACTOR_FIELDS),
                MissingValueFiller('Missing')
             ))]
        )
        fu.fit(missing_data)
        result = fu.transform(missing_data)
        exp_dict = {'a': [2, 2, 2, 4, 4, 4, 7, 8, 8, 8],
                    'b': ['123', '123', '123',
                          '234', '456', '456',
                          '789', '789', '789', '789'],
                    'c': [1.0, 2.0, 1.5, 4.0, 4.0, 4.0, 7.0, 9.0, 9.0, 9.0],
                    'd': ['a', 'a', 'Missing', 'Missing', 'e', 'f', 'Missing',
                          'h', 'j', 'j'],
                    'e': [1, 2, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
                    'f': ['a', 'b', 'Missing', 'Missing', 'Missing',
                          'Missing', 'Missing', 'Missing', 'Missing',
                          'Missing'],
                    'g': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'b',
                          'Missing'],
                    'h': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'Missing',
                          'Missing']
                    }
        expected = pd.DataFrame(exp_dict)
        expected = expected[['a', 'c', 'e', 'b', 'd', 'f', 'g', 'h']]
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_fit_transform(self, missing_data):
        # Test FeatureUnion fit_transform
        CONTINUOUS_FIELDS = missing_data.select_dtypes(
            ['int64', 'float64']).columns.tolist()
        FACTOR_FIELDS = missing_data.select_dtypes(['object']).columns
        CONTINUOUS_FIELDS.append('b')
        fu = FeatureUnion(
            [('Continuous Pipeline', make_pipeline(
                ColumnExtractor(CONTINUOUS_FIELDS),
                GroupByImputer('median', 'b'),
                ColumnDropper('b'),
                GroupByImputer('median')
             )),
             ('Factor Pipeline', make_pipeline(
                ColumnExtractor(FACTOR_FIELDS),
                MissingValueFiller('Missing')
             ))]
        )
        result = fu.fit_transform(missing_data)
        exp_dict = {'a': [2, 2, 2, 4, 4, 4, 7, 8, 8, 8],
                    'b': ['123', '123', '123',
                          '234', '456', '456',
                          '789', '789', '789', '789'],
                    'c': [1.0, 2.0, 1.5, 4.0, 4.0, 4.0, 7.0, 9.0, 9.0, 9.0],
                    'd': ['a', 'a', 'Missing', 'Missing', 'e', 'f', 'Missing',
                          'h', 'j', 'j'],
                    'e': [1, 2, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
                    'f': ['a', 'b', 'Missing', 'Missing', 'Missing',
                          'Missing', 'Missing', 'Missing', 'Missing',
                          'Missing'],
                    'g': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'b',
                          'Missing'],
                    'h': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'Missing',
                          'Missing']
                    }
        expected = pd.DataFrame(exp_dict)
        expected = expected[['a', 'c', 'e', 'b', 'd', 'f', 'g', 'h']]
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_unordered_index(self, missing_data):
        # Test FeatureUnion
        new_index = list(missing_data.index)
        new_index = new_index[::-1]
        missing_data.index = new_index

        CONTINUOUS_FIELDS = missing_data.select_dtypes(
            ['int64', 'float64']).columns.tolist()
        FACTOR_FIELDS = missing_data.select_dtypes(['object']).columns
        CONTINUOUS_FIELDS.append('b')
        fu = FeatureUnion(
            [('Continuous Pipeline', make_pipeline(
                ColumnExtractor(CONTINUOUS_FIELDS),
                GroupByImputer('median', 'b'),
                ColumnDropper('b'),
                GroupByImputer('median')
            )),
             ('Factor Pipeline', make_pipeline(
                 ColumnExtractor(FACTOR_FIELDS),
                 MissingValueFiller('Missing')
             ))]
        )
        fu.fit(missing_data)
        result = fu.transform(missing_data)
        exp_dict = {'a': [2, 2, 2, 4, 4, 4, 7, 8, 8, 8],
                    'b': ['123', '123', '123',
                          '234', '456', '456',
                          '789', '789', '789', '789'],
                    'c': [1.0, 2.0, 1.5, 4.0, 4.0, 4.0, 7.0, 9.0, 9.0, 9.0],
                    'd': ['a', 'a', 'Missing', 'Missing', 'e', 'f', 'Missing',
                          'h', 'j', 'j'],
                    'e': [1, 2, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
                    'f': ['a', 'b', 'Missing', 'Missing', 'Missing',
                          'Missing', 'Missing', 'Missing', 'Missing',
                          'Missing'],
                    'g': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'b',
                          'Missing'],
                    'h': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'Missing',
                          'Missing']
                    }
        expected = pd.DataFrame(exp_dict, index=new_index)
        expected = expected[['a', 'c', 'e', 'b', 'd', 'f', 'g', 'h']]
        tm.assert_frame_equal(result, expected, check_dtype=False)


@pytest.mark.usefixtures("missing_data")
class TestMakeUnion(object):

    def test_make_union(self, missing_data):
        # Test make_union
        CONTINUOUS_FIELDS = missing_data.select_dtypes(
            ['int64', 'float64']).columns.tolist()
        FACTOR_FIELDS = missing_data.select_dtypes(['object']).columns
        CONTINUOUS_FIELDS.append('b')
        fu = make_union(
            make_pipeline(
                ColumnExtractor(CONTINUOUS_FIELDS),
                GroupByImputer('median', 'b'),
                ColumnDropper('b'),
                GroupByImputer('median')
            ),
            make_pipeline(
                ColumnExtractor(FACTOR_FIELDS),
                MissingValueFiller('Missing')
            )
        )
        fu.fit(missing_data)
        result = fu.transform(missing_data)
        exp_dict = {'a': [2, 2, 2, 4, 4, 4, 7, 8, 8, 8],
                    'b': ['123', '123', '123',
                          '234', '456', '456',
                          '789', '789', '789', '789'],
                    'c': [1.0, 2.0, 1.5, 4.0, 4.0, 4.0, 7.0, 9.0, 9.0, 9.0],
                    'd': ['a', 'a', 'Missing', 'Missing', 'e', 'f', 'Missing',
                          'h', 'j', 'j'],
                    'e': [1, 2, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
                    'f': ['a', 'b', 'Missing', 'Missing', 'Missing',
                          'Missing', 'Missing', 'Missing', 'Missing',
                          'Missing'],
                    'g': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'b',
                          'Missing'],
                    'h': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'Missing',
                          'Missing']
                    }
        expected = pd.DataFrame(exp_dict)
        expected = expected[['a', 'c', 'e', 'b', 'd', 'f', 'g', 'h']]
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_value_error_kwargs(self, missing_data):
        # Test throwing value error because of kwargs
        CONTINUOUS_FIELDS = missing_data.select_dtypes(
            ['int64', 'float64']).columns.tolist()
        FACTOR_FIELDS = missing_data.select_dtypes(['object']).columns
        CONTINUOUS_FIELDS.append('b')
        with pytest.raises(TypeError):
            fu = make_union(
                make_pipeline(
                    ColumnExtractor(CONTINUOUS_FIELDS),
                    GroupByImputer('median', 'b'),
                    ColumnDropper('b'),
                    GroupByImputer('median')
                ),
                make_pipeline(
                    ColumnExtractor(FACTOR_FIELDS),
                    MissingValueFiller('Missing')
                ),
                **{'parameter': 'anything'}
            )
            fu
