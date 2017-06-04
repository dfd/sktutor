import pytest
import pandas as pd


@pytest.fixture
def missing_data():
    """Sample data for grouping by 1 column
    """
    data_dict = {'a': [2, 2, None, None, 4, 4, 7, 8, None, 8],
                 'b': ['123', '123', '123',
                       '234', '456', '456',
                       '789', '789', '789', '789'],
                 'c': [1, 2, None, 4, 4, 4, 7, 9, None, 9],
                 'd': ['a', 'a', None, None, 'e', 'f', None, 'h', 'j', 'j'],
                 'e': [1, 2, None, None, None, None, None, None, None, None],
                 'f': ['a', 'b', None, None, None, None, None, None, None,
                       None],
                 'g': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'b', None],
                 'h': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', None, None]
                 }
    df = pd.DataFrame(data_dict)
    return df


@pytest.fixture
def missing_data2():
    """Sample data for grouping by 2 columns
    """
    data_dict = {'a': [1, 2, None, None, 4, 4, 7, 8, None, 8],
                 'b': ['123', '123', '123',
                       '123', '123', '789',
                       '789', '789', '789', '789'],
                 'c': ['a', 'a', 'a', 'b', 'b', 'c', 'c', 'a', 'a', 'c'],
                 'd': ['a', 'a', None, None, 'e', 'f', None, 'h', 'j', 'j']
                 }
    df = pd.DataFrame(data_dict)
    return df


@pytest.fixture
def missing_data_factors():
    """DataFrame with missing factors data
    """
    data_dict = {'c': ['a', None, 'a', 'b', 'b', None, 'c', 'a', 'a', 'c'],
                 'd': ['a', 'a', None, None, 'e', 'f', None, 'h', 'j', 'j']
                 }
    df = pd.DataFrame(data_dict)
    return df


@pytest.fixture
def missing_data_numeric():
    """DataFrame with missing numberic data
    """
    data_dict = {'a': [2, 2, None, None, 4, 4, 7, 8, None, 8],
                 'c': [1, 2, None, 4, 4, 4, 7, 9, None, 9],
                 'e': [1, 2, None, None, None, None, None, None, None, None]
                 }
    df = pd.DataFrame(data_dict)
    return df


@pytest.fixture
def full_data_factors():
    """DataFrame with no missing factors values
    """
    data_dict = {'c': ['a', 'a', 'a', 'b', 'b', 'c', 'c', 'a', 'a', 'c'],
                 'd': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'j']
                 }
    df = pd.DataFrame(data_dict)
    return df


@pytest.fixture
def single_values_data():
    """DataFrame with single values in colums
    """
    data_dict = {'a': [2, 2, 2, 3, 4, 4, 7, 8, 8, 8],
                 'b': ['123', '123', '123',
                       '234', '456', '456',
                       '789', '789', '789', '789'],
                 'c': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 'd': [1, 1, 1, 1, 1, 1, 1, 1, 1, None],
                 'e': [1, 2, None, None, None, None, None, None, None, None],
                 'f': [None, None, None, None, None, None, None, None, None,
                       None],
                 'g': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', None],
                 'h': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a']
                 }
    df = pd.DataFrame(data_dict)
    return df


@pytest.fixture
def text_data():
    """DataFrame with text data
    """
    data_dict = {'a': ['Happy Birthday!', 'It\'s your  bday!'],
                 'b': ['Happy Arbor Day!', 'Happy Gilmore'],
                 'c': ['a', 'b']
                 }
    df = pd.DataFrame(data_dict)
    return df


@pytest.fixture
def boolean_data():
    """Sample boolean data for bitwise operators
    """
    data_dict = {'a': [True, True, False, False],
                 'b': [True, False, False, True],
                 'c': [False, True, True, False],
                 'd': [True, False, True, False],
                 'e': [False, True, False, True]
                 }
    df = pd.DataFrame(data_dict)
    return df


@pytest.fixture
def binary_data():
    """Sample binary data for bitwise operators
    """
    data_dict = {'a': [1, 1, 0, 0],
                 'b': [1, 0, 0, 1],
                 'c': [0, 1, 1, 0],
                 'd': [1, 0, 1, 0],
                 'e': [0, 1, 0, 1]
                 }
    df = pd.DataFrame(data_dict)
    return df


@pytest.fixture
def binary_series():
    return pd.Series([1, 1, 0, 0])
