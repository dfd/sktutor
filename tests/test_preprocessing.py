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
                                   ValueReplacer, FactorLimiter)
import pandas as pd
import pandas.util.testing as tm


@pytest.mark.usefixtures("missing_data")
@pytest.mark.usefixtures("missing_data2")
class TestGroupByImputer(object):
    """Sample pytest test function with the pytest fixture as an argument.
    """

    def test_groups_most_frequent(self, missing_data):
        gbi = GroupByImputer('most_frequent', 'b')
        gbi.fit(missing_data)
        result = gbi.transform(missing_data)
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
        gbi = GroupByImputer('mean', 'b')
        gbi.fit(missing_data)
        result = gbi.transform(missing_data)
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
        gbi = GroupByImputer('median', 'b')
        gbi.fit(missing_data)
        result = gbi.transform(missing_data)
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
        gbi = GroupByImputer('most_frequent')
        gbi.fit(missing_data)
        result = gbi.transform(missing_data)
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
        gbi = GroupByImputer('mean')
        gbi.fit(missing_data)
        result = gbi.transform(missing_data)
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
        gbi = GroupByImputer('median')
        gbi.fit(missing_data)
        result = gbi.transform(missing_data)
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
        gbi = GroupByImputer('stdev')
        with pytest.raises(ValueError):
            gbi.fit(missing_data)

    def test_key_error(self, missing_data):
        gbi = GroupByImputer('mean', 'b')
        gbi.fit(missing_data)
        exp_dict = {'a': [2, 2, None, None, 4, 4, 7, 8, None, 8],
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
        new_data = pd.DataFrame(exp_dict)
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
        result = gbi.transform(new_data)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_2groups_most_frequent(self, missing_data2):
        gbi = GroupByImputer('most_frequent', ['b', 'c'])
        gbi.fit(missing_data2)
        result = gbi.transform(missing_data2)
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
        gbi = GroupByImputer('mean', ['b', 'c'])
        gbi.fit(missing_data2)
        result = gbi.transform(missing_data2)
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
        gbi = GroupByImputer('median', ['b', 'c'])
        gbi.fit(missing_data2)
        result = gbi.transform(missing_data2)
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
    """Sample pytest test function with the pytest fixture as an argument.
    """

    def test_missing_factors(self, missing_data_factors):
        mvf = MissingValueFiller('Missing')
        result = mvf.fit_transform(missing_data_factors)
        exp_dict = {'c': ['a', 'Missing', 'a', 'b', 'b', 'Missing', 'c', 'a',
                          'a', 'c'],
                    'd': ['a', 'a', 'Missing', 'Missing', 'e', 'f', 'Missing',
                          'h', 'j', 'j']
                    }
        expected = pd.DataFrame(exp_dict)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_missing_numeric(self, missing_data_numeric):
        mvf = MissingValueFiller(0)
        result = mvf.fit_transform(missing_data_numeric)
        exp_dict = {'a': [2, 2, 0, 0, 4, 4, 7, 8, 0, 8],
                    'c': [1, 2, 0, 4, 4, 4, 7, 9, 0, 9],
                    'e': [1, 2, 0, 0, 0, 0, 0, 0, 0, 0]
                    }
        expected = pd.DataFrame(exp_dict)
        tm.assert_frame_equal(result, expected, check_dtype=False)


@pytest.mark.usefixtures("missing_data")
class TestOverMissingThresholdDropper(object):
    """Sample pytest test function with the pytest fixture as an argument.
    """

    def test_drop_20(self, missing_data):
        omtd = OverMissingThresholdDropper(.2)
        omtd.fit(missing_data)
        result = omtd.transform(missing_data)
        exp_dict = {'b': ['123', '123', '123',
                          '234', '456', '456',
                          '789', '789', '789', '789'],
                    'c': [1, 2, None, 4, 4, 4, 7, 9, None, 9],
                    'g': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'b', None],
                    'h': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', None, None]
                    }
        expected = pd.DataFrame(exp_dict)
        tm.assert_frame_equal(result, expected, check_dtype=False)


@pytest.mark.usefixtures("full_data_factors")
class TestValueReplacer(object):
    """Sample pytest test function with the pytest fixture as an argument.
    """

    def test_mapper(self, full_data_factors):
        mapper = {'c': {'a': 'z', 'b': 'z'},
                  'd': {'a': 'z', 'b': 'z', 'c': 'y', 'd': 'y', 'e': 'x',
                        'f': 'x', 'g': 'w', 'h': 'w', 'j': 'w'
                        }
                  }
        vr = ValueReplacer(mapper)
        vr.fit(full_data_factors)
        result = vr.transform(full_data_factors)
        exp_dict = {'c': ['z', 'z', 'z', 'z', 'z', 'c', 'c', 'z', 'z', 'c'],
                    'd': ['z', 'z', 'y', 'y', 'x', 'x', 'w', 'w', 'w', 'w']
                    }
        expected = pd.DataFrame(exp_dict)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_inverse_mapper(self, full_data_factors):
        inv_mapper = {'c': {'z': ['a', 'b']},
                      'd': {'z': ['a', 'b'],
                            'y': ['c', 'd'],
                            'x': ['e', 'f'],
                            'w': ['g', 'h', 'j']
                            }
                      }
        vr = ValueReplacer(inverse_mapper=inv_mapper)
        vr.fit(full_data_factors)
        result = vr.transform(full_data_factors)
        exp_dict = {'c': ['z', 'z', 'z', 'z', 'z', 'c', 'c', 'z', 'z', 'c'],
                    'd': ['z', 'z', 'y', 'y', 'x', 'x', 'w', 'w', 'w', 'w']
                    }
        expected = pd.DataFrame(exp_dict)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_extra_column_value_error(self, full_data_factors):
        mapper = {'c': {'a': 'z', 'b': 'z'},
                  'e': {'a': 'z', 'b': 'z', 'c': 'y', 'd': 'y', 'e': 'x',
                        'f': 'x', 'g': 'w', 'h': 'w', 'j': 'w'
                        }
                  }
        vr = ValueReplacer(mapper)
        with pytest.raises(ValueError):
            vr.fit(full_data_factors)

    def test_2_mappers_value_error(self):
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
            vr = ValueReplacer(mapper=mapper, inverse_mapper=inv_mapper)
            vr

    def test_no_mappers_value_error(self):
        with pytest.raises(ValueError):
            vr = ValueReplacer()
            vr


@pytest.mark.usefixtures("missing_data_factors")
class TestFactorLimiter(object):
    """Test the FactorLimiter class
    """

    def test_limiter(self, missing_data_factors):
        factors = {'c': {'factors': ['a', 'b'],
                         'default': 'a'
                         },
                   'd': {'factors': ['a', 'b', 'c', 'd'],
                         'default': 'd'
                         }
                   }
        fl = FactorLimiter(factors)
        fl.fit(missing_data_factors)
        result = fl.transform(missing_data_factors)
        exp_dict = {'c': ['a', 'a', 'a', 'b', 'b', 'a', 'a', 'a', 'a', 'a'],
                    'd': ['a', 'a', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd']
                    }
        expected = pd.DataFrame(exp_dict)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_extra_column_value_error(self, missing_data_factors):
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
