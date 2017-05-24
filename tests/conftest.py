import pytest
import pandas as pd


@pytest.fixture
def example_data():
    """Sample pytest fixture.
    See more at: http://doc.pytest.org/en/latest/fixture.html
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
