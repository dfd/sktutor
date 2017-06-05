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
                                   BitwiseOrApplicator, BitwiseAndApplicator)
import pandas as pd
import pandas.util.testing as tm


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


@pytest.mark.usefixtures("full_data_factors")
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


@pytest.mark.usefixtures("boolean_data")
class TestBitwiseOrApplicator(object):

    def test_mapper_boolean(self, boolean_data):
        # Test bitwise or applied to booleans
        mapper = {'f': ['c', 'd', 'e'],
                  'g': ['a', 'b']
                  }

        prep = BitwiseOrApplicator(mapper)
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

    def test_mapper_binary(self, boolean_data):
        # Test bitwise or applied to integers
        mapper = {'f': ['c', 'd', 'e'],
                  'g': ['a', 'b']
                  }

        prep = BitwiseOrApplicator(mapper)
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

    def test_extra_column_value_error(self, text_data):
        # Test throwing error when replacing values with a non-existant column.
        mapper = {'f': ['c', 'd', 'e'],
                  'g': ['a', 'b']
                  }
        prep = BitwiseOrApplicator(mapper)
        with pytest.raises(ValueError):
            prep.fit(text_data)


@pytest.mark.usefixtures("boolean_data")
class TestBitwiseAndApplicator(object):

    def test_mapper_boolean(self, boolean_data):
        # Test bitwise and applied to booleans
        mapper = {'f': ['c', 'd', 'e'],
                  'g': ['a', 'b']
                  }

        prep = BitwiseAndApplicator(mapper)
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

    def test_mapper_binary(self, boolean_data):
        # Test bitwise and applied to integers
        mapper = {'f': ['c', 'd', 'e'],
                  'g': ['a', 'b']
                  }

        prep = BitwiseAndApplicator(mapper)
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

    def test_extra_column_value_error(self, text_data):
        # Test throwing error when replacing values with a non-existant column.
        mapper = {'f': ['c', 'd', 'e'],
                  'g': ['a', 'b']
                  }
        prep = BitwiseAndApplicator(mapper)
        with pytest.raises(ValueError):
            prep.fit(text_data)
