#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_preprocessing
----------------------------------

Tests for `preprocessing` module.
"""

import pytest
from sktutor.preprocessing import (GroupByImputer, MissingValueFiller,
                                   OverMissingThresholdDropper,
                                   ValueReplacer, FactorLimiter,
                                   SingleValueAboveThresholdDropper,
                                   SingleValueDropper, ColumnExtractor,
                                   ColumnDropper, DummyCreator,
                                   ColumnValidator, TextContainsDummyExtractor,
                                   BitwiseOperator, BoxCoxTransformer,
                                   InteractionCreator, StandardScaler,
                                   PolynomialFeatures, ContinuousFeatureBinner,
                                   TypeExtractor, GenericTransformer,
                                   MissingColumnsReplacer)
from sktutor.pipeline import make_union
import numpy as np
import pandas as pd
import pandas.util.testing as tm
from random import shuffle
from sklearn.pipeline import make_pipeline


@pytest.mark.usefixtures("missing_data")
@pytest.mark.usefixtures("missing_data2")
class TestGroupByImputer(object):

    def test_groups_most_frequent(self, missing_data):
        # Test imputing most frequent value per group.
        prep = GroupByImputer('most_frequent', 'b')
        prep.fit(missing_data)
        result = prep.transform(missing_data)
        exp_dict = {'a': [2, 2, 2, None, 4, 4, 7, 8, 8, 8],
                    'b': ['123', '123', '123',
                          '234', '456', '456',
                          '789', '789', '789', '789'],
                    'c': [1.0, 2.0, 1.0, 4.0, 4.0, 4.0, 7.0, 9.0, 9.0, 9.0],
                    'd': ['a', 'a', 'a', None, 'e', 'f', 'j', 'h', 'j', 'j'],
                    'e': [1, 2, 1, None, None, None, None, None, None, None],
                    'f': ['a', 'b', 'a', None, None, None, None, None, None,
                          None],
                    'g': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'b', 'a'],
                    'h': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a']
                    }
        expected = pd.DataFrame(exp_dict)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_groups_mean(self, missing_data):
        # Test imputing mean by group.
        prep = GroupByImputer('mean', 'b')
        prep.fit(missing_data)
        result = prep.transform(missing_data)
        exp_dict = {'a': [2, 2, 2, None, 4, 4, 7, 8, 7 + 2/3, 8],
                    'b': ['123', '123', '123',
                          '234', '456', '456',
                          '789', '789', '789', '789'],
                    'c': [1.0, 2.0, 1.5, 4.0, 4.0, 4.0, 7.0, 9.0, 8+1/3, 9.0],
                    'd': ['a', 'a', None, None, 'e', 'f', None, 'h', 'j', 'j'],
                    'e': [1, 2, 1.5, None, None, None, None, None, None, None],
                    'f': ['a', 'b', None, None, None, None, None, None, None,
                          None],
                    'g': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'b', None],
                    'h': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', None, None]
                    }
        expected = pd.DataFrame(exp_dict)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_groups_median(self, missing_data):
        # Test imputing median by group.
        prep = GroupByImputer('median', 'b')
        prep.fit(missing_data)
        result = prep.transform(missing_data)
        exp_dict = {'a': [2, 2, 2, None, 4, 4, 7, 8, 8, 8],
                    'b': ['123', '123', '123',
                          '234', '456', '456',
                          '789', '789', '789', '789'],
                    'c': [1, 2, 1.5, 4, 4, 4, 7, 9, 9, 9],
                    'd': ['a', 'a', None, None, 'e', 'f', None, 'h', 'j', 'j'],
                    'e': [1, 2, 1.5, None, None, None, None, None, None, None],
                    'f': ['a', 'b', None, None, None, None, None, None, None,
                          None],
                    'g': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'b', None],
                    'h': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', None, None]
                    }
        expected = pd.DataFrame(exp_dict)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_all_most_frequent(self, missing_data):
        # Test imputing most frequent with no group by.
        prep = GroupByImputer('most_frequent')
        prep.fit(missing_data)
        result = prep.transform(missing_data)
        exp_dict = {'a': [2, 2, 2, 2, 4, 4, 7, 8, 2, 8],
                    'b': ['123', '123', '123',
                          '234', '456', '456',
                          '789', '789', '789', '789'],
                    'c': [1.0, 2.0, 4.0, 4.0, 4.0, 4.0, 7.0, 9.0, 4.0, 9.0],
                    'd': ['a', 'a', 'a', 'a', 'e', 'f', 'a', 'h', 'j', 'j'],
                    'e': [1, 2, 1, 1, 1, 1, 1, 1, 1, 1],
                    'f': ['a', 'b', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a'],
                    'g': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'b', 'a'],
                    'h': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a']
                    }
        expected = pd.DataFrame(exp_dict)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_all_mean(self, missing_data):
        # Test imputing mean with no group by.
        prep = GroupByImputer('mean')
        prep.fit(missing_data)
        result = prep.transform(missing_data)
        exp_dict = {'a': [2, 2, 5, 5, 4, 4, 7, 8, 5, 8],
                    'b': ['123', '123', '123',
                          '234', '456', '456',
                          '789', '789', '789', '789'],
                    'c': [1, 2, 5, 4, 4, 4, 7, 9, 5, 9],
                    'd': ['a', 'a', None, None, 'e', 'f', None, 'h', 'j', 'j'],
                    'e': [1, 2, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
                    'f': ['a', 'b', None, None, None, None, None, None, None,
                          None],
                    'g': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'b', None],
                    'h': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', None, None]
                    }
        expected = pd.DataFrame(exp_dict)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_all_median(self, missing_data):
        # Test imputing median with no group by.
        prep = GroupByImputer('median')
        prep.fit(missing_data)
        result = prep.transform(missing_data)
        exp_dict = {'a': [2, 2, 4, 4, 4, 4, 7, 8, 4, 8],
                    'b': ['123', '123', '123',
                          '234', '456', '456',
                          '789', '789', '789', '789'],
                    'c': [1, 2, 4, 4, 4, 4, 7, 9, 4, 9],
                    'd': ['a', 'a', None, None, 'e', 'f', None, 'h', 'j', 'j'],
                    'e': [1, 2, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
                    'f': ['a', 'b', None, None, None, None, None, None, None,
                          None],
                    'g': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'b', None],
                    'h': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', None, None]
                    }
        expected = pd.DataFrame(exp_dict)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_value_error(self, missing_data):
        # Test limiting options without a group by.
        prep = GroupByImputer('stdev')
        with pytest.raises(ValueError):
            prep.fit(missing_data)

    def test_key_error(self, missing_data):
        # Test imputing with np.nan when a new group level is introduced in
        # Transform.
        prep = GroupByImputer('mean', 'b')
        prep.fit(missing_data)
        new_dict = {'a': [2, 2, None, None, 4, 4, 7, 8, None, 8],
                    'b': ['123', '123', '123',
                          '987', '987', '456',
                          '789', '789', '789', '789'],
                    'c': [1, 2, None, 4, 4, 4, 7, 9, None, 9],
                    'd': ['a', 'a', None, None, 'e', 'f', None, 'h', 'j', 'j'],
                    'e': [1, 2, None, None, None, None, None, None, None,
                          None],
                    'f': ['a', 'b', None, None, None, None, None, None, None,
                          None],
                    'g': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'b', None],
                    'h': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', None, None]
                    }
        new_data = pd.DataFrame(new_dict)
        # set equal to the expected for test means group
        exp_dict = {'a': [2, 2, 2, None, 4, 4, 7, 8, 7+2/3, 8],
                    'b': ['123', '123', '123',
                          '987', '987', '456',
                          '789', '789', '789', '789'],
                    'c': [1.0, 2.0, 1.5, 4.0, 4.0, 4.0, 7.0, 9.0, 8+1/3, 9.0],
                    'd': ['a', 'a', None, None, 'e', 'f', None, 'h', 'j', 'j'],
                    'e': [1, 2, 1.5, None, None, None, None, None, None, None],
                    'f': ['a', 'b', None, None, None, None, None, None, None,
                          None],
                    'g': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'b', None],
                    'h': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', None, None]
                    }
        expected = pd.DataFrame(exp_dict)
        result = prep.transform(new_data)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_2groups_most_frequent(self, missing_data2):
        # Test most frequent with group by with 2 columns.
        prep = GroupByImputer('most_frequent', ['b', 'c'])
        prep.fit(missing_data2)
        result = prep.transform(missing_data2)
        exp_dict = {'a': [1, 2, 1, 4, 4, 4, 7, 8, 8, 8],
                    'b': ['123', '123', '123',
                          '123', '123', '789',
                          '789', '789', '789', '789'],
                    'c': ['a', 'a', 'a', 'b', 'b', 'c', 'c', 'a', 'a', 'c'],
                    'd': ['a', 'a', 'a', 'e', 'e', 'f', 'f', 'h', 'j', 'j']
                    }
        expected = pd.DataFrame(exp_dict)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_2groups_mean(self, missing_data2):
        # Test mean with group by with 2 columns.
        prep = GroupByImputer('mean', ['b', 'c'])
        prep.fit(missing_data2)
        result = prep.transform(missing_data2)
        exp_dict = {'a': [1, 2, 1.5, 4, 4, 4, 7, 8, 8, 8],
                    'b': ['123', '123', '123',
                          '123', '123', '789',
                          '789', '789', '789', '789'],
                    'c': ['a', 'a', 'a', 'b', 'b', 'c', 'c', 'a', 'a', 'c'],
                    'd': ['a', 'a', None, None, 'e', 'f', None, 'h', 'j',
                          'j']
                    }
        expected = pd.DataFrame(exp_dict)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_2groups_median(self, missing_data2):
        # Test median with group by with 2 columns.
        prep = GroupByImputer('median', ['b', 'c'])
        prep.fit(missing_data2)
        result = prep.transform(missing_data2)
        exp_dict = {'a': [1, 2, 1.5, 4, 4, 4, 7, 8, 8, 8],
                    'b': ['123', '123', '123',
                          '123', '123', '789',
                          '789', '789', '789', '789'],
                    'c': ['a', 'a', 'a', 'b', 'b', 'c', 'c', 'a', 'a', 'c'],
                    'd': ['a', 'a', None, None, 'e', 'f', None, 'h', 'j',
                          'j']
                    }
        expected = pd.DataFrame(exp_dict)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_unordered_index(self, missing_data):
        # Test unordered index is handled properly
        new_index = list(missing_data.index)
        shuffle(new_index)
        missing_data.index = new_index

        prep = GroupByImputer('most_frequent', 'b')
        prep.fit(missing_data)
        result = prep.transform(missing_data)
        exp_dict = {'a': [2, 2, 2, None, 4, 4, 7, 8, 8, 8],
                    'b': ['123', '123', '123',
                          '234', '456', '456',
                          '789', '789', '789', '789'],
                    'c': [1.0, 2.0, 1.0, 4.0, 4.0, 4.0, 7.0, 9.0, 9.0, 9.0],
                    'd': ['a', 'a', 'a', None, 'e', 'f', 'j', 'h', 'j', 'j'],
                    'e': [1, 2, 1, None, None, None, None, None, None, None],
                    'f': ['a', 'b', 'a', None, None, None, None, None, None,
                          None],
                    'g': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'b', 'a'],
                    'h': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a']
                    }
        expected = pd.DataFrame(exp_dict, index=new_index)
        tm.assert_frame_equal(result, expected, check_dtype=False)


@pytest.mark.usefixtures("missing_data_factors")
@pytest.mark.usefixtures("missing_data_numeric")
class TestMissingValueFiller(object):

    def test_missing_factors(self, missing_data_factors):
        # Test filling in missing factors with a string.
        prep = MissingValueFiller('Missing')
        result = prep.fit_transform(missing_data_factors)
        exp_dict = {'c': ['a', 'Missing', 'a', 'b', 'b', 'Missing', 'c', 'a',
                          'a', 'c'],
                    'd': ['a', 'a', 'Missing', 'Missing', 'e', 'f', 'Missing',
                          'h', 'j', 'j']
                    }
        expected = pd.DataFrame(exp_dict)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_missing_numeric(self, missing_data_numeric):
        # Test filling in missing numeric data with a number.
        prep = MissingValueFiller(0)
        result = prep.fit_transform(missing_data_numeric)
        exp_dict = {'a': [2, 2, 0, 0, 4, 4, 7, 8, 0, 8],
                    'c': [1, 2, 0, 4, 4, 4, 7, 9, 0, 9],
                    'e': [1, 2, 0, 0, 0, 0, 0, 0, 0, 0]
                    }
        expected = pd.DataFrame(exp_dict)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_unordered_index(self, missing_data_numeric):
        # Test unordered index is handled properly
        new_index = list(missing_data_numeric.index)
        shuffle(new_index)
        missing_data_numeric.index = new_index

        prep = MissingValueFiller(0)
        result = prep.fit_transform(missing_data_numeric)
        exp_dict = {'a': [2, 2, 0, 0, 4, 4, 7, 8, 0, 8],
                    'c': [1, 2, 0, 4, 4, 4, 7, 9, 0, 9],
                    'e': [1, 2, 0, 0, 0, 0, 0, 0, 0, 0]
                    }
        expected = pd.DataFrame(exp_dict, index=new_index)
        tm.assert_frame_equal(result, expected, check_dtype=False)


@pytest.mark.usefixtures("missing_data")
class TestOverMissingThresholdDropper(object):

    def test_drop_20(self, missing_data):
        # Test dropping columns with missing over a threshold.
        prep = OverMissingThresholdDropper(.2)
        prep.fit(missing_data)
        result = prep.transform(missing_data)
        exp_dict = {'b': ['123', '123', '123',
                          '234', '456', '456',
                          '789', '789', '789', '789'],
                    'c': [1, 2, None, 4, 4, 4, 7, 9, None, 9],
                    'g': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'b', None],
                    'h': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', None, None]
                    }
        expected = pd.DataFrame(exp_dict)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_threshold_high_value_error(self, missing_data):
        # Test throwing error with threshold set too high.
        with pytest.raises(ValueError):
            svatd = OverMissingThresholdDropper(1.5)
            svatd

    def test_threshold_low_value_error(self, missing_data):
        # Test throwing error with threshold set too low.
        with pytest.raises(ValueError):
            svatd = OverMissingThresholdDropper(-1)
            svatd

    def test_unordered_index(self, missing_data):
        # Test unordered index is handled properly
        new_index = list(missing_data.index)
        shuffle(new_index)
        missing_data.index = new_index

        prep = OverMissingThresholdDropper(.2)
        prep.fit(missing_data)
        result = prep.transform(missing_data)
        exp_dict = {'b': ['123', '123', '123',
                          '234', '456', '456',
                          '789', '789', '789', '789'],
                    'c': [1, 2, None, 4, 4, 4, 7, 9, None, 9],
                    'g': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'b', None],
                    'h': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', None, None]
                    }
        expected = pd.DataFrame(exp_dict, index=new_index)
        tm.assert_frame_equal(result, expected, check_dtype=False)


@pytest.mark.usefixtures("full_data_factors")
class TestValueReplacer(object):

    def test_mapper(self, full_data_factors):
        # Test replacing values with mapper.
        mapper = {'c': {'a': 'z', 'b': 'z'},
                  'd': {'a': 'z', 'b': 'z', 'c': 'y', 'd': 'y', 'e': 'x',
                        'f': 'x', 'g': 'w', 'h': 'w', 'j': 'w'
                        }
                  }
        prep = ValueReplacer(mapper)
        prep.fit(full_data_factors)
        result = prep.transform(full_data_factors)
        exp_dict = {'c': ['z', 'z', 'z', 'z', 'z', 'c', 'c', 'z', 'z', 'c'],
                    'd': ['z', 'z', 'y', 'y', 'x', 'x', 'w', 'w', 'w', 'w']
                    }
        expected = pd.DataFrame(exp_dict)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_inverse_mapper(self, full_data_factors):
        # Test replacing values with inverse_mapper.
        inv_mapper = {'c': {'z': ['a', 'b']},
                      'd': {'z': ['a', 'b'],
                            'y': ['c', 'd'],
                            'x': ['e', 'f'],
                            'w': ['g', 'h', 'j']
                            }
                      }
        prep = ValueReplacer(inverse_mapper=inv_mapper)
        prep.fit(full_data_factors)
        result = prep.transform(full_data_factors)
        exp_dict = {'c': ['z', 'z', 'z', 'z', 'z', 'c', 'c', 'z', 'z', 'c'],
                    'd': ['z', 'z', 'y', 'y', 'x', 'x', 'w', 'w', 'w', 'w']
                    }
        expected = pd.DataFrame(exp_dict)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_extra_column_value_error(self, full_data_factors):
        # Test throwing error when replacing values with a non-existant column.
        mapper = {'c': {'a': 'z', 'b': 'z'},
                  'e': {'a': 'z', 'b': 'z', 'c': 'y', 'd': 'y', 'e': 'x',
                        'f': 'x', 'g': 'w', 'h': 'w', 'j': 'w'
                        }
                  }
        prep = ValueReplacer(mapper)
        with pytest.raises(ValueError):
            prep.fit(full_data_factors)

    def test_2_mappers_value_error(self):
        # Test throwing error when specifying mapper and inverse_mapper.
        mapper = {'c': {'a': 'z', 'b': 'z'},
                  'e': {'a': 'z', 'b': 'z', 'c': 'y', 'd': 'y', 'e': 'x',
                        'f': 'x', 'g': 'w', 'h': 'w', 'j': 'w'
                        }
                  }
        inv_mapper = {'c': {'z': ['a', 'b']},
                      'd': {'z': ['a', 'b'],
                            'y': ['c', 'd'],
                            'x': ['e', 'f'],
                            'w': ['g', 'h', 'j']
                            }
                      }
        with pytest.raises(ValueError):
            prep = ValueReplacer(mapper=mapper, inverse_mapper=inv_mapper)
            prep

    def test_no_mappers_value_error(self):
        # Test throwing error when not specifying mapper or inverse_mapper.
        with pytest.raises(ValueError):
            prep = ValueReplacer()
            prep

    def test_unordered_index(self, full_data_factors):
        # Test unordered index is handled properly
        new_index = list(full_data_factors.index)
        shuffle(new_index)
        full_data_factors.index = new_index

        mapper = {'c': {'a': 'z', 'b': 'z'},
                  'd': {'a': 'z', 'b': 'z', 'c': 'y', 'd': 'y', 'e': 'x',
                        'f': 'x', 'g': 'w', 'h': 'w', 'j': 'w'
                        }
                  }
        prep = ValueReplacer(mapper)
        prep.fit(full_data_factors)
        result = prep.transform(full_data_factors)
        exp_dict = {'c': ['z', 'z', 'z', 'z', 'z', 'c', 'c', 'z', 'z', 'c'],
                    'd': ['z', 'z', 'y', 'y', 'x', 'x', 'w', 'w', 'w', 'w']
                    }
        expected = pd.DataFrame(exp_dict, index=new_index)
        tm.assert_frame_equal(result, expected, check_dtype=False)


@pytest.mark.usefixtures("missing_data_factors")
class TestFactorLimiter(object):

    def test_limiter(self, missing_data_factors):
        # Test limiting factor levels to specified levels with default.
        factors = {'c': {'factors': ['a', 'b'],
                         'default': 'a'
                         },
                   'd': {'factors': ['a', 'b', 'c', 'd'],
                         'default': 'd'
                         }
                   }
        prep = FactorLimiter(factors)
        prep.fit(missing_data_factors)
        result = prep.transform(missing_data_factors)
        exp_dict = {'c': ['a', 'a', 'a', 'b', 'b', 'a', 'a', 'a', 'a', 'a'],
                    'd': ['a', 'a', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd']
                    }
        expected = pd.DataFrame(exp_dict)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_extra_column_value_error(self, missing_data_factors):
        # Test throwing error when limiting values with a non-existant column.
        factors = {'c': {'factors': ['a', 'b'],
                         'default': 'a'
                         },
                   'e': {'factors': ['a', 'b', 'c', 'd'],
                         'default': 'd'
                         }
                   }
        fl = FactorLimiter(factors)
        with pytest.raises(ValueError):
            fl.fit(missing_data_factors)

    def test_unordered_index(self, missing_data_factors):
        # Test unordered index is handled properly
        new_index = list(missing_data_factors.index)
        shuffle(new_index)
        missing_data_factors.index = new_index

        factors = {'c': {'factors': ['a', 'b'],
                         'default': 'a'
                         },
                   'd': {'factors': ['a', 'b', 'c', 'd'],
                         'default': 'd'
                         }
                   }
        prep = FactorLimiter(factors)
        prep.fit(missing_data_factors)
        result = prep.transform(missing_data_factors)
        exp_dict = {'c': ['a', 'a', 'a', 'b', 'b', 'a', 'a', 'a', 'a', 'a'],
                    'd': ['a', 'a', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd']
                    }
        expected = pd.DataFrame(exp_dict, index=new_index)
        tm.assert_frame_equal(result, expected, check_dtype=False)


@pytest.mark.usefixtures("missing_data")
class TestSingleValueAboveThresholdDropper(object):

    def test_drop_70_with_na(self, missing_data):
        # test dropping columns with over 70% single value, including NaNs.
        prep = SingleValueAboveThresholdDropper(.7, dropna=False)
        prep.fit(missing_data)
        result = prep.transform(missing_data)
        exp_dict = {'a': [2, 2, None, None, 4, 4, 7, 8, None, 8],
                    'b': ['123', '123', '123',
                          '234', '456', '456',
                          '789', '789', '789', '789'],
                    'c': [1, 2, None, 4, 4, 4, 7, 9, None, 9],
                    'd': ['a', 'a', None, None, 'e', 'f', None, 'h', 'j', 'j']
                    }
        expected = pd.DataFrame(exp_dict)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_drop_70_without_na(self, missing_data):
        # test dropping columns with over 70% single value, not including NaNs.
        prep = SingleValueAboveThresholdDropper(.7, dropna=True)
        prep.fit(missing_data)
        result = prep.transform(missing_data)
        exp_dict = {'a': [2, 2, None, None, 4, 4, 7, 8, None, 8],
                    'b': ['123', '123', '123',
                          '234', '456', '456',
                          '789', '789', '789', '789'],
                    'c': [1, 2, None, 4, 4, 4, 7, 9, None, 9],
                    'd': ['a', 'a', None, None, 'e', 'f', None, 'h', 'j',
                          'j'],
                    'e': [1, 2, None, None, None, None, None, None, None,
                          None],
                    'f': ['a', 'b', None, None, None, None, None, None, None,
                          None],

                    }
        expected = pd.DataFrame(exp_dict)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_threshold_high_value_error(self, missing_data):
        # Test throwing error with threshold set too high.
        with pytest.raises(ValueError):
            prep = SingleValueAboveThresholdDropper(1.5)
            prep

    def test_threshold_low_value_error(self, missing_data):
        # Test throwing error with threshold set too low.
        with pytest.raises(ValueError):
            prep = SingleValueAboveThresholdDropper(-1)
            prep

    def test_unordered_index(self, missing_data):
        # Test unordered index is handled properly
        new_index = list(missing_data.index)
        shuffle(new_index)
        missing_data.index = new_index

        prep = SingleValueAboveThresholdDropper(.7, dropna=False)
        prep.fit(missing_data)
        result = prep.transform(missing_data)
        exp_dict = {'a': [2, 2, None, None, 4, 4, 7, 8, None, 8],
                    'b': ['123', '123', '123',
                          '234', '456', '456',
                          '789', '789', '789', '789'],
                    'c': [1, 2, None, 4, 4, 4, 7, 9, None, 9],
                    'd': ['a', 'a', None, None, 'e', 'f', None, 'h', 'j', 'j']
                    }
        expected = pd.DataFrame(exp_dict, index=new_index)
        tm.assert_frame_equal(result, expected, check_dtype=False)


@pytest.mark.usefixtures("single_values_data")
class TestSingleValueDropper(object):

    def test_without_na(self, single_values_data):
        # Test dropping columns with single values, excluding NaNs as a value.
        prep = SingleValueDropper(dropna=True)
        prep.fit(single_values_data)
        result = prep.transform(single_values_data)
        exp_dict = {'a': [2, 2, 2, 3, 4, 4, 7, 8, 8, 8],
                    'b': ['123', '123', '123',
                          '234', '456', '456',
                          '789', '789', '789', '789'],
                    'e': [1, 2, None, None, None, None, None, None, None,
                          None]
                    }
        expected = pd.DataFrame(exp_dict)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_with_na(self, single_values_data):
        # Test dropping columns with single values, including NaNs as a value.
        prep = SingleValueDropper(dropna=False)
        prep.fit(single_values_data)
        result = prep.transform(single_values_data)
        exp_dict = {'a': [2, 2, 2, 3, 4, 4, 7, 8, 8, 8],
                    'b': ['123', '123', '123',
                          '234', '456', '456',
                          '789', '789', '789', '789'],
                    'd': [1, 1, 1, 1, 1, 1, 1, 1, 1, None],
                    'e': [1, 2, None, None, None, None, None, None, None,
                          None],
                    'g': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', None]
                    }
        expected = pd.DataFrame(exp_dict)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_unordered_index(self, single_values_data):
        # Test unordered index is handled properly
        new_index = list(single_values_data.index)
        shuffle(new_index)
        single_values_data.index = new_index

        prep = SingleValueDropper(dropna=False)
        prep.fit(single_values_data)
        result = prep.transform(single_values_data)
        exp_dict = {'a': [2, 2, 2, 3, 4, 4, 7, 8, 8, 8],
                    'b': ['123', '123', '123',
                          '234', '456', '456',
                          '789', '789', '789', '789'],
                    'd': [1, 1, 1, 1, 1, 1, 1, 1, 1, None],
                    'e': [1, 2, None, None, None, None, None, None, None,
                          None],
                    'g': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', None]
                    }
        expected = pd.DataFrame(exp_dict, index=new_index)
        tm.assert_frame_equal(result, expected, check_dtype=False)


@pytest.mark.usefixtures("missing_data")
class TestColumnExtractor(object):

    def test_extraction(self, missing_data):
        # Test extraction of columns from a DataFrame.
        prep = ColumnExtractor(['a', 'b', 'c'])
        prep.fit(missing_data)
        result = prep.transform(missing_data)
        exp_dict = {'a': [2, 2, None, None, 4, 4, 7, 8, None, 8],
                    'b': ['123', '123', '123',
                          '234', '456', '456',
                          '789', '789', '789', '789'],
                    'c': [1, 2, None, 4, 4, 4, 7, 9, None, 9]
                    }
        expected = pd.DataFrame(exp_dict)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_column_missing_error(self, missing_data):
        # Test throwing error when an extraction is requested of a missing.
        # column
        prep = ColumnExtractor(['a', 'b', 'z'])
        with pytest.raises(ValueError):
            prep.fit(missing_data)

    def test_unordered_index(self, missing_data):
        # Test unordered index is handled properly
        new_index = list(missing_data.index)
        shuffle(new_index)
        missing_data.index = new_index

        prep = ColumnExtractor(['a', 'b', 'c'])
        prep.fit(missing_data)
        result = prep.transform(missing_data)
        exp_dict = {'a': [2, 2, None, None, 4, 4, 7, 8, None, 8],
                    'b': ['123', '123', '123',
                          '234', '456', '456',
                          '789', '789', '789', '789'],
                    'c': [1, 2, None, 4, 4, 4, 7, 9, None, 9]
                    }
        expected = pd.DataFrame(exp_dict, index=new_index)
        tm.assert_frame_equal(result, expected, check_dtype=False)


@pytest.mark.usefixtures("missing_data")
class TestColumnDropper(object):

    def test_drop_multiple(self, missing_data):
        # Test extraction of columns from a DataFrame
        prep = ColumnDropper(['d', 'e', 'f', 'g', 'h'])
        prep.fit(missing_data)
        result = prep.transform(missing_data)
        exp_dict = {'a': [2, 2, None, None, 4, 4, 7, 8, None, 8],
                    'b': ['123', '123', '123',
                          '234', '456', '456',
                          '789', '789', '789', '789'],
                    'c': [1, 2, None, 4, 4, 4, 7, 9, None, 9]
                    }
        expected = pd.DataFrame(exp_dict)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_drop_single(self, missing_data):
        # Test extraction of columns from a DataFrame
        prep = ColumnDropper('d')
        prep.fit(missing_data)
        result = prep.transform(missing_data)
        exp_dict = {'a': [2, 2, None, None, 4, 4, 7, 8, None, 8],
                    'b': ['123', '123', '123',
                          '234', '456', '456',
                          '789', '789', '789', '789'],
                    'c': [1, 2, None, 4, 4, 4, 7, 9, None, 9],
                    'e': [1, 2, None, None, None, None, None, None, None,
                          None],
                    'f': ['a', 'b', None, None, None, None, None, None, None,
                          None],
                    'g': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'b', None],
                    'h': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', None, None]
                    }
        expected = pd.DataFrame(exp_dict)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_error(self, missing_data):
        # Test throwing error when dropping is requested of a missing column
        prep = ColumnDropper(['a', 'b', 'z'])
        with pytest.raises(ValueError):
            prep.fit(missing_data)

    def test_unordered_index(self, missing_data):
        # Test unordered index is handled properly
        new_index = list(missing_data.index)
        shuffle(new_index)
        missing_data.index = new_index

        prep = ColumnDropper(['d', 'e', 'f', 'g', 'h'])
        prep.fit(missing_data)
        result = prep.transform(missing_data)
        exp_dict = {'a': [2, 2, None, None, 4, 4, 7, 8, None, 8],
                    'b': ['123', '123', '123',
                          '234', '456', '456',
                          '789', '789', '789', '789'],
                    'c': [1, 2, None, 4, 4, 4, 7, 9, None, 9]
                    }
        expected = pd.DataFrame(exp_dict, index=new_index)
        tm.assert_frame_equal(result, expected, check_dtype=False)


@pytest.mark.usefixtures("full_data_factors")
@pytest.mark.usefixtures("full_data_factors_subset")
@pytest.mark.usefixtures("missing_data_factors")
class TestDummyCreator(object):

    def test_default_dummies(self, full_data_factors):
        # Test creating dummies variables from a DataFrame
        prep = DummyCreator()
        prep.fit(full_data_factors)
        result = prep.transform(full_data_factors)
        exp_dict = {'c_a': [1, 1, 1, 0, 0, 0, 0, 1, 1, 0],
                    'c_b': [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                    'c_c': [0, 0, 0, 0, 0, 1, 1, 0, 0, 1],
                    'd_a': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    'd_b': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    'd_c': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    'd_d': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    'd_e': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    'd_f': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    'd_g': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                    'd_h': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    'd_j': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
                    }
        expected = pd.DataFrame(exp_dict)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_fit_transform(self, full_data_factors):
        # Test creating dummies variables from a DataFrame
        prep = DummyCreator()
        result = prep.fit_transform(full_data_factors)
        exp_dict = {'c_a': [1, 1, 1, 0, 0, 0, 0, 1, 1, 0],
                    'c_b': [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                    'c_c': [0, 0, 0, 0, 0, 1, 1, 0, 0, 1],
                    'd_a': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    'd_b': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    'd_c': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    'd_d': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    'd_e': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    'd_f': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    'd_g': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                    'd_h': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    'd_j': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
                    }
        expected = pd.DataFrame(exp_dict)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_drop_first_dummies(self, full_data_factors):
        # Test dropping first dummies for each column.
        kwargs = {'drop_first': True}
        prep = DummyCreator(**kwargs)
        prep.fit(full_data_factors)
        result = prep.transform(full_data_factors)
        exp_dict = {'c_b': [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                    'c_c': [0, 0, 0, 0, 0, 1, 1, 0, 0, 1],
                    'd_b': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    'd_c': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    'd_d': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    'd_e': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    'd_f': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    'd_g': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                    'd_h': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    'd_j': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
                    }
        expected = pd.DataFrame(exp_dict)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_drop_first_dummies_missing_levels(self, full_data_factors,
                                               full_data_factors_subset):
        # Test dropping first dummies for each column.
        kwargs = {'drop_first': True}
        prep = DummyCreator(**kwargs)
        prep.fit(full_data_factors)
        result = prep.transform(full_data_factors_subset)
        exp_dict = {'c_b': [1, 1, 0, 0, 0, 0, 0],
                    'c_c': [0, 0, 1, 1, 0, 0, 1],
                    'd_b': [0, 0, 0, 0, 0, 0, 0],
                    'd_c': [0, 0, 0, 0, 0, 0, 0],
                    'd_d': [1, 0, 0, 0, 0, 0, 0],
                    'd_e': [0, 1, 0, 0, 0, 0, 0],
                    'd_f': [0, 0, 1, 0, 0, 0, 0],
                    'd_g': [0, 0, 0, 1, 0, 0, 0],
                    'd_h': [0, 0, 0, 0, 1, 0, 0],
                    'd_j': [0, 0, 0, 0, 0, 1, 1]
                    }
        expected = pd.DataFrame(exp_dict)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_dummy_na_false_dummies(self, missing_data_factors):
        # Test not creating dummies for NaNs.
        prep = DummyCreator()
        prep.fit(missing_data_factors)
        result = prep.transform(missing_data_factors)
        exp_dict = {'c_a': [1, 0, 1, 0, 0, 0, 0, 1, 1, 0],
                    'c_b': [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                    'c_c': [0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                    'd_a': [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    'd_e': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    'd_f': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    'd_h': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    'd_j': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
                    }
        expected = pd.DataFrame(exp_dict)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_dummy_na_true_dummies(self, missing_data_factors):
        # Test creating dummies for NaNs.
        kwargs = {'dummy_na': True}
        prep = DummyCreator(**kwargs)
        prep.fit(missing_data_factors)
        result = prep.transform(missing_data_factors)
        exp_dict = {'c_a': [1, 0, 1, 0, 0, 0, 0, 1, 1, 0],
                    'c_b': [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                    'c_c': [0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                    'c_nan': [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                    'd_a': [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    'd_e': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    'd_f': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    'd_h': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    'd_j': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                    'd_nan': [0, 0, 1, 1, 0, 0, 1, 0, 0, 0]
                    }
        expected = pd.DataFrame(exp_dict)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_fillin_missing_dummies(self, full_data_factors):
        # Test filling missing dummies with a transform data missing levels
        # present in the fitting data set.
        prep = DummyCreator()
        prep.fit(full_data_factors)
        new_dict = {'c': ['b', 'c'],
                    'd': ['a', 'b']
                    }
        new_data = pd.DataFrame(new_dict)
        result = prep.transform(new_data)
        exp_dict = {'c_a': [0, 0],
                    'c_b': [1, 0],
                    'c_c': [0, 1],
                    'd_a': [1, 0],
                    'd_b': [0, 1],
                    'd_c': [0, 0],
                    'd_d': [0, 0],
                    'd_e': [0, 0],
                    'd_f': [0, 0],
                    'd_g': [0, 0],
                    'd_h': [0, 0],
                    'd_j': [0, 0]
                    }
        expected = pd.DataFrame(exp_dict)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_unordered_index(self, full_data_factors):
        # Test unordered index is handled properly
        new_index = list(full_data_factors.index)
        shuffle(new_index)
        full_data_factors.index = new_index

        prep = DummyCreator()
        prep.fit(full_data_factors)
        result = prep.transform(full_data_factors)
        exp_dict = {'c_a': [1, 1, 1, 0, 0, 0, 0, 1, 1, 0],
                    'c_b': [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                    'c_c': [0, 0, 0, 0, 0, 1, 1, 0, 0, 1],
                    'd_a': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    'd_b': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    'd_c': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    'd_d': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    'd_e': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    'd_f': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    'd_g': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                    'd_h': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    'd_j': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
                    }
        expected = pd.DataFrame(exp_dict, index=new_index)
        tm.assert_frame_equal(result, expected, check_dtype=False)


@pytest.mark.usefixtures("full_data_factors")
class TestColumnValidator(object):

    def test_order(self, full_data_factors):
        # Test extraction of columns from a DataFrame
        prep = ColumnValidator()
        prep.fit(full_data_factors)
        new_dict = {'d': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'j'],
                    'c': ['a', 'a', 'a', 'b', 'b', 'c', 'c', 'a', 'a', 'c']
                    }
        new_data = pd.DataFrame(new_dict)
        result = prep.transform(new_data)
        exp_dict = {'c': ['a', 'a', 'a', 'b', 'b', 'c', 'c', 'a', 'a', 'c'],
                    'd': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'j']
                    }
        expected = pd.DataFrame(exp_dict)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_missing_columns_error(self, full_data_factors):
        # Test throwing an error when the new data is missing columns
        prep = ColumnValidator()
        prep.fit(full_data_factors)
        new_dict = {'d': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'j']
                    }
        new_data = pd.DataFrame(new_dict)
        with pytest.raises(ValueError):
            prep.transform(new_data)

    def test_new_columns_error(self, full_data_factors):
        # Test throwing an error when the new data is missing columns
        prep = ColumnValidator()
        prep.fit(full_data_factors)
        new_dict = {'c': ['a', 'a', 'a', 'b', 'b', 'c', 'c', 'a', 'a', 'c'],
                    'd': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'j'],
                    'e': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'j']
                    }
        new_data = pd.DataFrame(new_dict)
        with pytest.raises(ValueError):
            prep.transform(new_data)

    def test_unordered_index(self, full_data_factors):
        # Test unordered index is handled properly
        new_index = list(full_data_factors.index)
        shuffle(new_index)
        full_data_factors.index = new_index

        prep = ColumnValidator()
        prep.fit(full_data_factors)
        result = prep.transform(full_data_factors)

        exp_dict = {
            'c': ['a', 'a', 'a', 'b', 'b', 'c', 'c', 'a', 'a', 'c'],
            'd': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'j']
        }

        expected = pd.DataFrame(exp_dict, index=new_index)
        tm.assert_frame_equal(result, expected, check_dtype=False)


@pytest.mark.usefixtures("text_data")
class TestTextContainsDummyExtractor(object):

    def test_mapper(self, text_data):
        # Test text contains dummy with mapper.
        mapper = {'a':
                  {'a_1':
                   [{'pattern': 'birthday', 'kwargs': {'case': False}},
                    {'pattern': 'bday', 'kwargs': {'case': False}}
                    ],
                   'a_2':
                   [{'pattern': 'b.*day', 'kwargs': {'case': False}}
                    ],
                   },
                  'b':
                  {'b_1':
                   [{'pattern': 'h.*r', 'kwargs': {'case': False}}
                    ],
                   'b_2':
                   [{'pattern': '!', 'kwargs': {'case': False}},
                    ]
                   }
                  }
        prep = TextContainsDummyExtractor(mapper)
        prep.fit(text_data)
        result = prep.transform(text_data)
        exp_dict = {'a': ['Happy Birthday!', 'It\'s your  bday!'],
                    'b': ['Happy Arbor Day!', 'Happy Gilmore'],
                    'c': ['a', 'b'],
                    'a_1': [1, 1],
                    'a_2': [1, 1],
                    'b_1': [1, 1],
                    'b_2': [1, 0]
                    }
        expected = pd.DataFrame(exp_dict)
        tm.assert_frame_equal(result, expected, check_dtype=False,
                              check_like=True)

    def test_extra_column_value_error(self, text_data):
        # Test throwing error when replacing values with a non-existant column.
        mapper = {'a':
                  {'a_1':
                   [{'pattern': 'birthday', 'kwargs': {'case': False}},
                    {'pattern': 'bday', 'kwargs': {'case': False}}
                    ],
                   'a_2':
                   [{'pattern': 'b.*day', 'kwargs': {'case': False}}
                    ],
                   },
                  'd':
                  {'b_1':
                   [{'pattern': 'h.*r', 'kwargs': {'case': False}}
                    ],
                   'b_2':
                   [{'pattern': '!', 'kwargs': {'case': False}},
                    ]
                   }
                  }
        prep = TextContainsDummyExtractor(mapper)
        with pytest.raises(ValueError):
            prep.fit(text_data)

    def test_unordered_index(self, text_data):
        # Test unordered index is handled properly
        new_index = list(text_data.index)
        shuffle(new_index)
        text_data.index = new_index

        mapper = {'a':
                  {'a_1':
                   [{'pattern': 'birthday', 'kwargs': {'case': False}},
                    {'pattern': 'bday', 'kwargs': {'case': False}}
                    ],
                   'a_2':
                   [{'pattern': 'b.*day', 'kwargs': {'case': False}}
                    ],
                   },
                  'b':
                  {'b_1':
                   [{'pattern': 'h.*r', 'kwargs': {'case': False}}
                    ],
                   'b_2':
                   [{'pattern': '!', 'kwargs': {'case': False}},
                    ]
                   }
                  }
        prep = TextContainsDummyExtractor(mapper)
        prep.fit(text_data)
        result = prep.transform(text_data)
        exp_dict = {'a': ['Happy Birthday!', 'It\'s your  bday!'],
                    'b': ['Happy Arbor Day!', 'Happy Gilmore'],
                    'c': ['a', 'b'],
                    'a_1': [1, 1],
                    'a_2': [1, 1],
                    'b_1': [1, 1],
                    'b_2': [1, 0]
                    }
        expected = pd.DataFrame(exp_dict, index=new_index)
        tm.assert_frame_equal(result, expected, check_dtype=False,
                              check_like=True)


@pytest.mark.usefixtures("boolean_data")
class TestBitwiseOperator(object):

    def test_operator_value_error(self, text_data):
        # Test throwing error when using invalid operator parameter
        mapper = {'f': ['c', 'd', 'e'],
                  'g': ['a', 'b']
                  }
        with pytest.raises(ValueError):
            prep = BitwiseOperator('with', mapper)
            prep

    def test_or_mapper_boolean(self, boolean_data):
        # Test bitwise or applied to booleans
        mapper = {'f': ['c', 'd', 'e'],
                  'g': ['a', 'b']
                  }

        prep = BitwiseOperator('or', mapper)
        prep.fit(boolean_data)
        result = prep.transform(boolean_data)
        exp_dict = {'a': [True, True, False, False],
                    'b': [True, False, False, True],
                    'c': [False, True, True, False],
                    'd': [True, False, True, False],
                    'e': [False, True, False, True],
                    'f': [1, 1, 1, 1],
                    'g': [1, 1, 0, 1],
                    }

        expected = pd.DataFrame(exp_dict)
        tm.assert_frame_equal(result, expected, check_dtype=False,
                              check_like=True)

    def test_or_mapper_binary(self, boolean_data):
        # Test bitwise or applied to integers
        mapper = {'f': ['c', 'd', 'e'],
                  'g': ['a', 'b']
                  }

        prep = BitwiseOperator('or', mapper)
        prep.fit(boolean_data)
        result = prep.transform(boolean_data)
        exp_dict = {'a': [1, 1, 0, 0],
                    'b': [1, 0, 0, 1],
                    'c': [0, 1, 1, 0],
                    'd': [1, 0, 1, 0],
                    'e': [0, 1, 0, 1],
                    'f': [1, 1, 1, 1],
                    'g': [1, 1, 0, 1],
                    }

        expected = pd.DataFrame(exp_dict)
        tm.assert_frame_equal(result, expected, check_dtype=False,
                              check_like=True)

    def test_or_extra_column_value_error(self, text_data):
        # Test throwing error when replacing values with a non-existant column.
        mapper = {'f': ['c', 'd', 'e'],
                  'g': ['a', 'b']
                  }
        prep = BitwiseOperator('or', mapper)
        with pytest.raises(ValueError):
            prep.fit(text_data)

    def test_and_mapper_boolean(self, boolean_data):
        # Test bitwise and applied to booleans
        mapper = {'f': ['c', 'd', 'e'],
                  'g': ['a', 'b']
                  }

        prep = BitwiseOperator('and', mapper)
        prep.fit(boolean_data)
        result = prep.transform(boolean_data)
        exp_dict = {'a': [True, True, False, False],
                    'b': [True, False, False, True],
                    'c': [False, True, True, False],
                    'd': [True, False, True, False],
                    'e': [False, True, False, True],
                    'f': [0, 0, 0, 0],
                    'g': [1, 0, 0, 0]
                    }

        expected = pd.DataFrame(exp_dict)
        tm.assert_frame_equal(result, expected, check_dtype=False,
                              check_like=True)

    def test_and_mapper_binary(self, boolean_data):
        # Test bitwise and applied to integers
        mapper = {'f': ['c', 'd', 'e'],
                  'g': ['a', 'b']
                  }

        prep = BitwiseOperator('and', mapper)
        prep.fit(boolean_data)
        result = prep.transform(boolean_data)
        exp_dict = {'a': [1, 1, 0, 0],
                    'b': [1, 0, 0, 1],
                    'c': [0, 1, 1, 0],
                    'd': [1, 0, 1, 0],
                    'e': [0, 1, 0, 1],
                    'f': [0, 0, 0, 0],
                    'g': [1, 0, 0, 0]
                    }

        expected = pd.DataFrame(exp_dict)
        tm.assert_frame_equal(result, expected, check_dtype=False,
                              check_like=True)

    def test_and_extra_column_value_error(self, text_data):
        # Test throwing error when replacing values with a non-existant column.
        mapper = {'f': ['c', 'd', 'e'],
                  'g': ['a', 'b']
                  }
        prep = BitwiseOperator('and', mapper)
        with pytest.raises(ValueError):
            prep.fit(text_data)

    def test_unordered_index(self, boolean_data):
        # Test unordered index is handled properly
        new_index = list(boolean_data.index)
        shuffle(new_index)
        boolean_data.index = new_index

        mapper = {
            'f': ['c', 'd', 'e'],
            'g': ['a', 'b']
        }

        prep = BitwiseOperator('or', mapper)
        prep.fit(boolean_data)
        result = prep.transform(boolean_data)
        exp_dict = {'a': [True, True, False, False],
                    'b': [True, False, False, True],
                    'c': [False, True, True, False],
                    'd': [True, False, True, False],
                    'e': [False, True, False, True],
                    'f': [1, 1, 1, 1],
                    'g': [1, 1, 0, 1],
                    }

        expected = pd.DataFrame(exp_dict, index=new_index)
        tm.assert_frame_equal(result, expected, check_dtype=False,
                              check_like=True)


@pytest.mark.usefixtures("full_data_numeric")
class TestBoxCoxTransformer(object):

    def test_fit_transfrom(self, full_data_numeric):
        # test default functionalty
        prep = BoxCoxTransformer()
        result = prep.fit_transform(full_data_numeric)
        exp_dict = {'a': [0.71695113, 0.71695113, 0.71695113,
                          1.15921005, 1.48370246, 1.48370246,
                          2.1414305, 2.30371316, 2.30371316,
                          2.30371316],
                    'c': [0., 0.8310186, 1.47159953, 2.0132148,
                          2.0132148, 2.0132148, 3.32332097, 4.0444457,
                          4.0444457, 4.0444457],
                    'e': [0., 0.89952678, 1.67649211, 2.38322965,
                          3.04195191, 3.66477648, 4.25925117,
                          4.83048775,  5.38215505,  5.91700138]
                    }
        expected = pd.DataFrame(exp_dict)
        tm.assert_frame_equal(result, expected, check_dtype=False,
                              check_like=True)

    def test_fit_then_transform(self, full_data_numeric):
        # test using fit then transform
        prep = BoxCoxTransformer()
        prep.fit(full_data_numeric)
        result = prep.transform(full_data_numeric)
        exp_dict = {'a': [0.71695113, 0.71695113, 0.71695113,
                          1.15921005, 1.48370246, 1.48370246,
                          2.1414305, 2.30371316, 2.30371316,
                          2.30371316],
                    'c': [0., 0.8310186, 1.47159953, 2.0132148,
                          2.0132148, 2.0132148, 3.32332097, 4.0444457,
                          4.0444457, 4.0444457],
                    'e': [0., 0.89952678, 1.67649211, 2.38322965,
                          3.04195191, 3.66477648, 4.25925117,
                          4.83048775,  5.38215505,  5.91700138]
                    }
        expected = pd.DataFrame(exp_dict)
        tm.assert_frame_equal(result, expected, check_dtype=False,
                              check_like=True)

    def test_unordered_index(self, full_data_numeric):
        # Test unordered index is handled properly
        new_index = list(full_data_numeric.index)
        shuffle(new_index)
        full_data_numeric.index = new_index

        prep = BoxCoxTransformer()
        result = prep.fit_transform(full_data_numeric)
        exp_dict = {'a': [0.71695113, 0.71695113, 0.71695113,
                          1.15921005, 1.48370246, 1.48370246,
                          2.1414305, 2.30371316, 2.30371316,
                          2.30371316],
                    'c': [0., 0.8310186, 1.47159953, 2.0132148,
                          2.0132148, 2.0132148, 3.32332097, 4.0444457,
                          4.0444457, 4.0444457],
                    'e': [0., 0.89952678, 1.67649211, 2.38322965,
                          3.04195191, 3.66477648, 4.25925117,
                          4.83048775,  5.38215505,  5.91700138]
                    }
        expected = pd.DataFrame(exp_dict, index=new_index)
        tm.assert_frame_equal(result, expected, check_dtype=False,
                              check_like=True)


@pytest.mark.usefixtures("interaction_data")
class TestInteractionCreator(object):

    def test_interactions(self, interaction_data):
        # test generation of interactions
        prep = InteractionCreator(columns1=['a', 'b'],
                                  columns2=['c', 'd', 'e'])
        result = prep.fit_transform(interaction_data)
        exp_dict = {'a': [2, 3, 4, 5],
                    'b': [1, 0, 0, 1],
                    'c': [0, 1, 1, 0],
                    'd': [1, 0, 1, 0],
                    'e': [0, 1, 0, 1],
                    'a:c': [0, 3, 4, 0],
                    'a:d': [2, 0, 4, 0],
                    'a:e': [0, 3, 0, 5],
                    'b:c': [0, 0, 0, 0],
                    'b:d': [1, 0, 0, 0],
                    'b:e': [0, 0, 0, 1]
                    }
        expected = pd.DataFrame(exp_dict)
        print(result)
        tm.assert_frame_equal(result, expected, check_dtype=False,
                              check_like=True)

    def test__extra_column_value_error(self, interaction_data):
        # test value error with non-existent columns
        prep = InteractionCreator(columns1=['a', 'f'],
                                  columns2=['c', 'd', 'g'])

        with pytest.raises(ValueError):
            prep.fit_transform(interaction_data)

    def test_unordered_index(self, interaction_data):
        # Test unordered index is handled properly
        new_index = list(interaction_data.index)
        shuffle(new_index)
        interaction_data.index = new_index

        prep = InteractionCreator(columns1=['a', 'b'],
                                  columns2=['c', 'd', 'e'])
        result = prep.fit_transform(interaction_data)
        exp_dict = {'a': [2, 3, 4, 5],
                    'b': [1, 0, 0, 1],
                    'c': [0, 1, 1, 0],
                    'd': [1, 0, 1, 0],
                    'e': [0, 1, 0, 1],
                    'a:c': [0, 3, 4, 0],
                    'a:d': [2, 0, 4, 0],
                    'a:e': [0, 3, 0, 5],
                    'b:c': [0, 0, 0, 0],
                    'b:d': [1, 0, 0, 0],
                    'b:e': [0, 0, 0, 1]
                    }
        expected = pd.DataFrame(exp_dict, index=new_index)
        print(result)
        tm.assert_frame_equal(result, expected, check_dtype=False,
                              check_like=True)


@pytest.mark.usefixtures("full_data_numeric")
class TestStandardScaler(object):

    def test_fit_transform(self, full_data_numeric):
        # test default functionalty
        prep = StandardScaler()
        result = prep.fit_transform(full_data_numeric)
        exp_dict = {'a': [-1.11027222, -1.11027222, -1.11027222, -0.71374643,
                          -0.31722063, -0.31722063,  0.87235674,  1.26888254,
                          1.26888254,  1.26888254],
                    'c': [-1.45260037, -1.10674314, -0.76088591, -0.41502868,
                          -0.41502868, -0.41502868,  0.62254302,  1.31425748,
                          1.31425748,  1.31425748],
                    'e': [-1.5666989, -1.21854359, -0.87038828, -0.52223297,
                          -0.17407766, 0.17407766, 0.52223297, 0.87038828,
                          1.21854359, 1.5666989]
                    }
        expected = pd.DataFrame(exp_dict)
        tm.assert_frame_equal(result, expected, check_dtype=False,
                              check_like=True)

    def test_fit_transform_defined_columns(self, full_data_numeric):
        # test defining which columns to apply standardization to
        prep = StandardScaler(columns=['a', 'e'])
        result = prep.fit_transform(full_data_numeric)
        exp_dict = {
            'a': [-1.11027222, -1.11027222, -1.11027222, -0.71374643,
                  -0.31722063, -0.31722063,  0.87235674,  1.26888254,
                  1.26888254,  1.26888254],
            'c': [1, 2, 3, 4, 4, 4, 7, 9, 9, 9],
            'e': [-1.5666989, -1.21854359, -0.87038828, -0.52223297,
                  -0.17407766, 0.17407766, 0.52223297, 0.87038828,
                  1.21854359, 1.5666989]
        }
        expected = pd.DataFrame(exp_dict)
        tm.assert_frame_equal(result, expected, check_dtype=False,
                              check_like=True)

    def test_fit_then_transform(self, full_data_numeric):
        # test using fit then transform
        prep = StandardScaler()
        prep.fit(full_data_numeric)
        result = prep.transform(full_data_numeric)
        exp_dict = {'a': [-1.11027222, -1.11027222, -1.11027222, -0.71374643,
                          -0.31722063, -0.31722063,  0.87235674,  1.26888254,
                          1.26888254,  1.26888254],
                    'c': [-1.45260037, -1.10674314, -0.76088591, -0.41502868,
                          -0.41502868, -0.41502868,  0.62254302,  1.31425748,
                          1.31425748,  1.31425748],
                    'e': [-1.5666989, -1.21854359, -0.87038828, -0.52223297,
                          -0.17407766, 0.17407766, 0.52223297, 0.87038828,
                          1.21854359, 1.5666989]
                    }
        expected = pd.DataFrame(exp_dict)
        tm.assert_frame_equal(result, expected, check_dtype=False,
                              check_like=True)

    def test_fit_then_transform_defined_columns(self, full_data_numeric):
        # test defining which columns to apply standardization to
        prep = StandardScaler(columns=['a', 'e'])
        prep.fit(full_data_numeric)
        result = prep.transform(full_data_numeric)
        exp_dict = {
            'a': [-1.11027222, -1.11027222, -1.11027222, -0.71374643,
                  -0.31722063, -0.31722063,  0.87235674,  1.26888254,
                  1.26888254,  1.26888254],
            'c': [1, 2, 3, 4, 4, 4, 7, 9, 9, 9],
            'e': [-1.5666989, -1.21854359, -0.87038828, -0.52223297,
                  -0.17407766, 0.17407766, 0.52223297, 0.87038828,
                  1.21854359, 1.5666989]
        }
        expected = pd.DataFrame(exp_dict)
        tm.assert_frame_equal(result, expected, check_dtype=False,
                              check_like=True)

    def test_fit_then_partial_transform(self, full_data_numeric):
        # test using fit then transform on specified columns
        prep = StandardScaler()
        prep.fit(full_data_numeric)
        result = prep.transform(X=full_data_numeric, partial_cols=['c', 'e'])
        exp_dict = {'a': [-1.11027222, -1.11027222, -1.11027222, -0.71374643,
                          -0.31722063, -0.31722063,  0.87235674,  1.26888254,
                          1.26888254,  1.26888254],
                    'c': [-1.45260037, -1.10674314, -0.76088591, -0.41502868,
                          -0.41502868, -0.41502868,  0.62254302,  1.31425748,
                          1.31425748,  1.31425748],
                    'e': [-1.5666989, -1.21854359, -0.87038828, -0.52223297,
                          -0.17407766, 0.17407766, 0.52223297, 0.87038828,
                          1.21854359, 1.5666989]
                    }
        expected = pd.DataFrame(exp_dict)
        expected = expected[['c', 'e']]
        tm.assert_frame_equal(result, expected, check_dtype=False,
                              check_like=True)

    def test_unordered_index(self, full_data_numeric):
        # Test unordered index is handled properly
        new_index = list(full_data_numeric.index)
        shuffle(new_index)
        full_data_numeric.index = new_index

        prep = StandardScaler()
        prep.fit(full_data_numeric)
        result = prep.transform(full_data_numeric)
        exp_dict = {'a': [-1.11027222, -1.11027222, -1.11027222, -0.71374643,
                          -0.31722063, -0.31722063, 0.87235674, 1.26888254,
                          1.26888254, 1.26888254],
                    'c': [-1.45260037, -1.10674314, -0.76088591, -0.41502868,
                          -0.41502868, -0.41502868, 0.62254302, 1.31425748,
                          1.31425748, 1.31425748],
                    'e': [-1.5666989, -1.21854359, -0.87038828, -0.52223297,
                          -0.17407766, 0.17407766, 0.52223297, 0.87038828,
                          1.21854359, 1.5666989],
                    }
        expected = pd.DataFrame(exp_dict, index=new_index)
        tm.assert_frame_equal(result, expected, check_dtype=False,
                              check_like=False)

    def test_inverse_transform(self, full_data_numeric):
        # test inverse_transform
        new_index = list(full_data_numeric.index)
        shuffle(new_index)
        full_data_numeric.index = new_index

        prep = StandardScaler()
        transformed = prep.fit_transform(full_data_numeric)
        original = prep.inverse_transform(transformed)

        tm.assert_frame_equal(
            full_data_numeric,
            original,
            check_dtype=False,
            check_like=True
        )

    def test_inverse_partial_transform(self, full_data_numeric):
        # test inverse_transform
        new_index = list(full_data_numeric.index)
        shuffle(new_index)
        full_data_numeric.index = new_index

        prep = StandardScaler()
        transformed = prep.fit_transform(full_data_numeric)
        partial_original = prep.inverse_transform(
            transformed, partial_cols=['a', 'e']
        )

        tm.assert_frame_equal(
            full_data_numeric[['a', 'e']],
            partial_original,
            check_dtype=False,
            check_like=True
        )

    def test_inverse_transform_defined_columns(self, full_data_numeric):
        # test defining which columns to apply standardization to
        prep = StandardScaler(columns=['a', 'e'])
        prep.fit(full_data_numeric)
        transformed = prep.fit_transform(full_data_numeric)
        result = prep.inverse_transform(transformed)
        tm.assert_frame_equal(
            result, full_data_numeric, check_dtype=False, check_like=True
        )


@pytest.mark.usefixtures("full_data_numeric")
class TestPolynomialFeatures(object):

    def test_polynomial_features(self, full_data_numeric):
        # test polynomial feature creation
        prep = PolynomialFeatures(degree=3)

        result = prep.fit_transform(full_data_numeric)
        exp_dict = {
            'a': [2, 2, 2, 3, 4, 4, 7, 8, 8, 8],
            'c': [1, 2, 3, 4, 4, 4, 7, 9, 9, 9],
            'e': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'a^2': [4, 4, 4, 9, 16, 16, 49, 64, 64, 64],
            'a*c': [2, 4, 6, 12, 16, 16, 49, 72, 72, 72],
            'a*e': [2, 4, 6, 12, 20, 24, 49, 64, 72, 80],
            'c^2': [1, 4, 9, 16, 16, 16, 49, 81, 81, 81],
            'c*e': [1, 4, 9, 16, 20, 24, 49, 72, 81, 90],
            'e^2': [1, 4, 9, 16, 25, 36, 49, 64, 81, 100],
            'a^3': [8, 8, 8, 27, 64, 64, 343, 512, 512, 512],
            'a^2*c': [4, 8, 12, 36, 64, 64, 343, 576, 576, 576],
            'a^2*e': [4, 8, 12, 36, 80, 96, 343, 512, 576, 640],
            'a*c^2': [2, 8, 18, 48, 64, 64, 343, 648, 648, 648],
            'a*c*e': [2, 8, 18, 48, 80, 96, 343, 576, 648, 720],
            'a*e^2': [2, 8, 18, 48, 100, 144, 343, 512, 648, 800],
            'c^3': [1, 8, 27, 64, 64, 64, 343, 729, 729, 729],
            'c^2*e': [1, 8, 27, 64, 80, 96, 343, 648, 729, 810],
            'c*e^2': [1, 8, 27, 64, 100, 144, 343, 576, 729, 900],
            'e^3': [1, 8, 27, 64, 125, 216, 343, 512, 729, 1000]
        }
        expected = pd.DataFrame(exp_dict)
        expected = expected[[
            'a', 'c', 'e', 'a^2', 'a*c', 'a*e',
            'c^2', 'c*e', 'e^2', 'a^3', 'a^2*c',
            'a^2*e', 'a*c^2', 'a*c*e', 'a*e^2',
            'c^3', 'c^2*e', 'c*e^2', 'e^3'
        ]]

        tm.assert_frame_equal(
            result,
            expected,
            check_dtype=False,
        )

    def test_polynomial_features_interactions(self, full_data_numeric):
        # test polynomial feature creation
        prep = PolynomialFeatures(interaction_only=True)

        result = prep.fit_transform(full_data_numeric)
        exp_dict = {
            'a': [2, 2, 2, 3, 4, 4, 7, 8, 8, 8],
            'c': [1, 2, 3, 4, 4, 4, 7, 9, 9, 9],
            'e': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'a*c': [2, 4, 6, 12, 16, 16, 49, 72, 72, 72],
            'a*e': [2, 4, 6, 12, 20, 24, 49, 64, 72, 80],
            'c*e': [1, 4, 9, 16, 20, 24, 49, 72, 81, 90],
        }
        expected = pd.DataFrame(exp_dict)
        expected = expected[[
            'a', 'c', 'e', 'a*c', 'a*e', 'c*e'
        ]]

        tm.assert_frame_equal(
            result,
            expected,
            check_dtype=False,
        )

    def test_unordered_index(self, full_data_numeric):
        # Test unordered index is handled properly
        new_index = list(full_data_numeric.index)
        shuffle(new_index)
        full_data_numeric.index = new_index

        # test polynomial feature creation
        prep = PolynomialFeatures(degree=3)

        result = prep.fit_transform(full_data_numeric)
        exp_dict = {
            'a': [2, 2, 2, 3, 4, 4, 7, 8, 8, 8],
            'c': [1, 2, 3, 4, 4, 4, 7, 9, 9, 9],
            'e': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'a^2': [4, 4, 4, 9, 16, 16, 49, 64, 64, 64],
            'a*c': [2, 4, 6, 12, 16, 16, 49, 72, 72, 72],
            'a*e': [2, 4, 6, 12, 20, 24, 49, 64, 72, 80],
            'c^2': [1, 4, 9, 16, 16, 16, 49, 81, 81, 81],
            'c*e': [1, 4, 9, 16, 20, 24, 49, 72, 81, 90],
            'e^2': [1, 4, 9, 16, 25, 36, 49, 64, 81, 100],
            'a^3': [8, 8, 8, 27, 64, 64, 343, 512, 512, 512],
            'a^2*c': [4, 8, 12, 36, 64, 64, 343, 576, 576, 576],
            'a^2*e': [4, 8, 12, 36, 80, 96, 343, 512, 576, 640],
            'a*c^2': [2, 8, 18, 48, 64, 64, 343, 648, 648, 648],
            'a*c*e': [2, 8, 18, 48, 80, 96, 343, 576, 648, 720],
            'a*e^2': [2, 8, 18, 48, 100, 144, 343, 512, 648, 800],
            'c^3': [1, 8, 27, 64, 64, 64, 343, 729, 729, 729],
            'c^2*e': [1, 8, 27, 64, 80, 96, 343, 648, 729, 810],
            'c*e^2': [1, 8, 27, 64, 100, 144, 343, 576, 729, 900],
            'e^3': [1, 8, 27, 64, 125, 216, 343, 512, 729, 1000]
        }
        expected = pd.DataFrame(exp_dict, index=new_index)
        expected = expected[[
            'a', 'c', 'e', 'a^2', 'a*c', 'a*e',
            'c^2', 'c*e', 'e^2', 'a^3', 'a^2*c',
            'a^2*e', 'a*c^2', 'a*c*e', 'a*e^2',
            'c^3', 'c^2*e', 'c*e^2', 'e^3'
        ]]

        tm.assert_frame_equal(
            result,
            expected,
            check_dtype=False,
        )


@pytest.mark.usefixtures("missing_data")
class TestContinuousFeatureBinner(object):

    def test_feature_binning(self, missing_data):
        # test standard use
        prep = ContinuousFeatureBinner(
            field='a',
            bins=[0, 3, 6, 9]
        )
        result = prep.fit_transform(missing_data)

        expected = {
            'a': [2, 2, None, None, 4, 4, 7, 8, None, 8],
            'b': ['123', '123', '123', '234', '456',
                  '456', '789', '789', '789', '789'],
            'c': [1, 2, None, 4, 4, 4, 7, 9, None, 9],
            'd': ['a', 'a', None, None, 'e', 'f', None, 'h', 'j', 'j'],
            'e': [1, 2, None, None, None, None, None, None, None, None],
            'f': ['a', 'b', None, None, None, None, None, None, None, None],
            'g': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'b', None],
            'h': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', None, None],
            'a_GRP': ['(0.0, 3.0]', '(0.0, 3.0]', 'Other', 'Other',
                      '(3.0, 6.0]', '(3.0, 6.0]', '(6.0, 9.0]',
                      '(6.0, 9.0]', 'Other', '(6.0, 9.0]']
        }
        expected = pd.DataFrame(expected, index=missing_data.index)
        expected = expected[['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'a_GRP']]

        tm.assert_frame_equal(
            result,
            expected,
            check_dtype=False,
        )

    def test_missing_field_error(self, missing_data):
        # test that specifying a field that doesn't exist returns error
        prep = ContinuousFeatureBinner(
            field='x',
            bins=[0, 3, 6, 9]
        )

        with pytest.raises(ValueError):
            prep.fit_transform(missing_data)

    def test_feature_binning_right(self, missing_data):
        # test standard use
        prep = ContinuousFeatureBinner(
            field='a',
            bins=[0, 4, 8],
            right_inclusive=False
        )
        result = prep.fit_transform(missing_data)
        expected = {
            'a': [2, 2, None, None, 4, 4, 7, 8, None, 8],
            'b': ['123', '123', '123', '234', '456',
                  '456', '789', '789', '789', '789'],
            'c': [1, 2, None, 4, 4, 4, 7, 9, None, 9],
            'd': ['a', 'a', None, None, 'e', 'f', None, 'h', 'j', 'j'],
            'e': [1, 2, None, None, None, None, None, None, None, None],
            'f': ['a', 'b', None, None, None, None, None, None, None, None],
            'g': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'b', None],
            'h': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', None, None],
            'a_GRP': ['[0.0, 4.0)', '[0.0, 4.0)', 'Other', 'Other',
                      '[4.0, 8.0)', '[4.0, 8.0)', '[4.0, 8.0)', 'Other',
                      'Other', 'Other']
        }
        expected = pd.DataFrame(expected, index=missing_data.index)
        expected = expected[['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'a_GRP']]

        tm.assert_frame_equal(
            result,
            expected,
            check_dtype=False,
        )

    def test_feature_binning_range(self, missing_data):
        # test use with range()
        prep = ContinuousFeatureBinner(
            field='a',
            bins=range(10)
        )
        result = prep.fit_transform(missing_data)

        expected = {
            'a': [2, 2, None, None, 4, 4, 7, 8, None, 8],
            'b': ['123', '123', '123', '234', '456',
                  '456', '789', '789', '789', '789'],
            'c': [1, 2, None, 4, 4, 4, 7, 9, None, 9],
            'd': ['a', 'a', None, None, 'e', 'f', None, 'h', 'j', 'j'],
            'e': [1, 2, None, None, None, None, None, None, None, None],
            'f': ['a', 'b', None, None, None, None, None, None, None, None],
            'g': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'b', None],
            'h': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', None, None],
            'a_GRP': ['(1.0, 2.0]', '(1.0, 2.0]', 'Other', 'Other',
                      '(3.0, 4.0]', '(3.0, 4.0]', '(6.0, 7.0]',
                      '(7.0, 8.0]', 'Other', '(7.0, 8.0]']
        }
        expected = pd.DataFrame(expected, index=missing_data.index)
        expected = expected[['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'a_GRP']]

        tm.assert_frame_equal(
            result,
            expected,
            check_dtype=False,
        )

    def test_feature_binning_numpy_range(self, missing_data):
        # test standard use
        prep = ContinuousFeatureBinner(
            field='a',
            bins=np.arange(0, 10, 3)
        )
        result = prep.fit_transform(missing_data)

        expected = {
            'a': [2, 2, None, None, 4, 4, 7, 8, None, 8],
            'b': ['123', '123', '123', '234', '456',
                  '456', '789', '789', '789', '789'],
            'c': [1, 2, None, 4, 4, 4, 7, 9, None, 9],
            'd': ['a', 'a', None, None, 'e', 'f', None, 'h', 'j', 'j'],
            'e': [1, 2, None, None, None, None, None, None, None, None],
            'f': ['a', 'b', None, None, None, None, None, None, None, None],
            'g': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'b', None],
            'h': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', None, None],
            'a_GRP': ['(0.0, 3.0]', '(0.0, 3.0]', 'Other', 'Other',
                      '(3.0, 6.0]', '(3.0, 6.0]', '(6.0, 9.0]',
                      '(6.0, 9.0]', 'Other', '(6.0, 9.0]']
        }
        expected = pd.DataFrame(expected, index=missing_data.index)
        expected = expected[['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'a_GRP']]

        tm.assert_frame_equal(
            result,
            expected,
            check_dtype=False,
        )

    def test_unordered_index(self, missing_data):
        # Test unordered index is handled properly
        new_index = list(missing_data.index)
        shuffle(new_index)
        missing_data.index = new_index

        # test standard use
        prep = ContinuousFeatureBinner(
            field='a',
            bins=[0, 3, 6, 9]
        )
        result = prep.fit_transform(missing_data)

        expected = {
            'a': [2, 2, None, None, 4, 4, 7, 8, None, 8],
            'b': ['123', '123', '123', '234', '456',
                  '456', '789', '789', '789', '789'],
            'c': [1, 2, None, 4, 4, 4, 7, 9, None, 9],
            'd': ['a', 'a', None, None, 'e', 'f', None, 'h', 'j', 'j'],
            'e': [1, 2, None, None, None, None, None, None, None, None],
            'f': ['a', 'b', None, None, None, None, None, None, None, None],
            'g': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'b', None],
            'h': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', None, None],
            'a_GRP': ['(0.0, 3.0]', '(0.0, 3.0]', 'Other', 'Other',
                      '(3.0, 6.0]', '(3.0, 6.0]', '(6.0, 9.0]',
                      '(6.0, 9.0]', 'Other', '(6.0, 9.0]']
        }
        expected = pd.DataFrame(expected, index=missing_data.index)
        expected = expected[['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'a_GRP']]

        tm.assert_frame_equal(
            result,
            expected,
            check_dtype=False,
        )


@pytest.mark.usefixtures("missing_data")
class TestTypeExtractor(object):

    def test_type_extractor_numeric(self, missing_data):
        # test standard use
        prep = TypeExtractor('numeric')
        result = prep.fit_transform(missing_data)

        expected = {
            'a': [2, 2, None, None, 4, 4, 7, 8, None, 8],
            'c': [1, 2, None, 4, 4, 4, 7, 9, None, 9],
            'e': [1, 2, None, None, None, None, None, None, None, None],
        }
        expected = pd.DataFrame(expected, index=missing_data.index)
        expected = expected[['a', 'c', 'e']]

        tm.assert_frame_equal(
            result,
            expected,
            check_dtype=False,
        )

    def test_type_extractor_categorical(self, missing_data):
        # test standard use
        prep = TypeExtractor('categorical')
        result = prep.fit_transform(missing_data)

        expected = {
            'b': ['123', '123', '123', '234', '456',
                  '456', '789', '789', '789', '789'],
            'd': ['a', 'a', None, None, 'e', 'f', None, 'h', 'j', 'j'],
            'f': ['a', 'b', None, None, None, None, None, None, None, None],
            'g': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'b', None],
            'h': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', None, None],
        }
        expected = pd.DataFrame(expected, index=missing_data.index)
        expected = expected[['b', 'd', 'f', 'g', 'h']]

        tm.assert_frame_equal(
            result,
            expected,
            check_dtype=False,
        )

    def test_type_extractor_feature_union(self, missing_data):
        # test in typical use case with FeatureUnion()
        fu = make_union(
            make_pipeline(
                TypeExtractor('numeric'),
            ),
            make_pipeline(
                TypeExtractor('categorical'),
            )
        )
        result = fu.fit_transform(missing_data)

        exp_dict = {
            'a': [2, 2, None, None, 4, 4, 7, 8, None, 8],
            'c': [1, 2, None, 4, 4, 4, 7, 9, None, 9],
            'e': [1, 2, None, None, None, None, None, None, None, None],
            'b': ['123', '123', '123', '234', '456',
                  '456', '789', '789', '789', '789'],
            'd': ['a', 'a', None, None, 'e', 'f', None, 'h', 'j', 'j'],
            'f': ['a', 'b', None, None, None, None, None, None, None, None],
            'g': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'b', None],
            'h': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', None, None],
        }
        expected = pd.DataFrame(exp_dict)
        expected = expected[['a', 'c', 'e', 'b', 'd', 'f', 'g', 'h']]

        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_type_extractor_feature_union_none(self, missing_data):
        # test in typical use case with FeatureUnion() with one dtype
        # without results
        missing_data = missing_data[['a', 'c', 'e']]

        fu = make_union(
            make_pipeline(
                TypeExtractor('numeric'),
            ),
            make_pipeline(
                TypeExtractor('categorical'),
            )
        )
        result = fu.fit_transform(missing_data)

        exp_dict = {
            'a': [2, 2, None, None, 4, 4, 7, 8, None, 8],
            'c': [1, 2, None, 4, 4, 4, 7, 9, None, 9],
            'e': [1, 2, None, None, None, None, None, None, None, None],
        }
        expected = pd.DataFrame(exp_dict)
        expected = expected[['a', 'c', 'e']]

        tm.assert_frame_equal(result, expected, check_dtype=False)


@pytest.mark.usefixtures("missing_data2")
class TestGenericTransformer(object):

    def test_generic_transformer(self, missing_data2):
        # test simple function use
        def add_bias(df):
            df['bias'] = 1
            return df

        prep = GenericTransformer(add_bias)
        result = prep.fit_transform(missing_data2)

        data_dict = {
            'a': [1, 2, None, None, 4, 4, 7, 8, None, 8],
            'b': ['123', '123', '123', '123', '123', '789',
                  '789', '789', '789', '789'],
            'c': ['a', 'a', 'a', 'b', 'b', 'c', 'c', 'a', 'a', 'c'],
            'd': ['a', 'a', None, None, 'e', 'f', None, 'h', 'j', 'j'],
            'bias': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        }
        expected = pd.DataFrame(data_dict)
        expected = expected[['a', 'b', 'c', 'd', 'bias']]

        tm.assert_frame_equal(
            result,
            expected,
            check_dtype=False,
        )

    def test_generic_transformer_pipe_parameters(self, missing_data2):
        # test with function that takes parameters in a pipeline
        def add_bias(df):
            df['bias'] = 1
            return df

        def add_columns(df, col1, col2):
            df['new_col'] = df[col1] + df[col2]
            return df

        prep = make_pipeline(
            GenericTransformer(add_bias),
            GenericTransformer(
                function=add_columns,
                params={'col1': 'a', 'col2': 'bias'}
            )
        )
        result = prep.fit_transform(missing_data2)

        data_dict = {
            'a': [1, 2, None, None, 4, 4, 7, 8, None, 8],
            'b': ['123', '123', '123', '123', '123', '789',
                  '789', '789', '789', '789'],
            'c': ['a', 'a', 'a', 'b', 'b', 'c', 'c', 'a', 'a', 'c'],
            'd': ['a', 'a', None, None, 'e', 'f', None, 'h', 'j', 'j'],
            'bias': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'new_col': [2, 3, None, None, 5, 5, 8, 9, None, 9]
        }
        expected = pd.DataFrame(data_dict)
        expected = expected[['a', 'b', 'c', 'd', 'bias', 'new_col']]

        tm.assert_frame_equal(
            result,
            expected,
            check_dtype=False,
        )

    def test_generic_transformer_unordered(self, missing_data2):
        # Test unordered index is handled properly
        new_index = list(missing_data2.index)
        shuffle(new_index)
        missing_data2.index = new_index

        def add_bias(df):
            df['bias'] = 1
            return df

        def add_columns(df, col1, col2):
            df['new_col'] = df[col1] + df[col2]
            return df

        prep = make_pipeline(
            GenericTransformer(add_bias),
            GenericTransformer(
                function=add_columns,
                params={'col1': 'a', 'col2': 'bias'}
            )
        )
        result = prep.fit_transform(missing_data2)

        data_dict = {
            'a': [1, 2, None, None, 4, 4, 7, 8, None, 8],
            'b': ['123', '123', '123', '123', '123', '789',
                  '789', '789', '789', '789'],
            'c': ['a', 'a', 'a', 'b', 'b', 'c', 'c', 'a', 'a', 'c'],
            'd': ['a', 'a', None, None, 'e', 'f', None, 'h', 'j', 'j'],
            'bias': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'new_col': [2, 3, None, None, 5, 5, 8, 9, None, 9]
        }
        expected = pd.DataFrame(data_dict, index=new_index)
        expected = expected[['a', 'b', 'c', 'd', 'bias', 'new_col']]

        tm.assert_frame_equal(
            result,
            expected,
            check_dtype=False,
        )


@pytest.mark.usefixtures("full_data_numeric")
class TestMissingColumnsReplacer(object):

    def test_missing_transformer(self, full_data_numeric):
        # test two missing colums
        cols = ['a', 'b', 'c', 'd', 'e']

        prep = MissingColumnsReplacer(cols, 0)
        result = prep.fit_transform(full_data_numeric)

        data_dict = {'a': [2, 2, 2, 3, 4, 4, 7, 8, 8, 8],
                     'c': [1, 2, 3, 4, 4, 4, 7, 9, 9, 9],
                     'e': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                     'b': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     'd': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                     }
        expected = pd.DataFrame(data_dict).reindex_axis(
            ['a', 'c', 'e', 'b', 'd'], axis=1)

        tm.assert_frame_equal(
            result,
            expected,
            check_dtype=False,
        )

    def test_missing_transformer_none_missing(self, full_data_numeric):
        # test no missing colums
        cols = ['a', 'c', 'e']

        prep = MissingColumnsReplacer(cols, 0)
        result = prep.fit_transform(full_data_numeric)

        data_dict = {'a': [2, 2, 2, 3, 4, 4, 7, 8, 8, 8],
                     'c': [1, 2, 3, 4, 4, 4, 7, 9, 9, 9],
                     'e': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                     }
        expected = pd.DataFrame(data_dict)

        tm.assert_frame_equal(
            result,
            expected,
            check_dtype=False,
        )

    def test_missing_transformer_unordered(self, full_data_numeric):
        # Test unordered index is handled properly
        new_index = sorted(list(full_data_numeric.index), reverse=True)
        full_data_numeric.index = new_index

        cols = ['a', 'b', 'c', 'd', 'e']

        prep = MissingColumnsReplacer(cols, 0)
        result = prep.fit_transform(full_data_numeric)

        data_dict = {'a': [2, 2, 2, 3, 4, 4, 7, 8, 8, 8],
                     'c': [1, 2, 3, 4, 4, 4, 7, 9, 9, 9],
                     'e': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                     'b': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     'd': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                     }
        expected = pd.DataFrame(data_dict, index=new_index).reindex_axis(
            ['a', 'c', 'e', 'b', 'd'], axis=1)

        tm.assert_frame_equal(
            result,
            expected,
            check_dtype=False,
        )
