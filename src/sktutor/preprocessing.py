# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    StandardScaler as ScikitStandardScaler,
    PolynomialFeatures as ScikitPolynomialFeatures
)
import numpy as np
from sktutor.utils import dict_with_default, dict_default, bitwise_operator
from scipy import stats
from patsy import dmatrix
import re
from collections import OrderedDict


def mode(x):
    """Return the most frequent occurance.  If two or more values are tied
    with the most occurances, then return the lowest value.

    :param x: A data vector.
    :type x: pandas Series
    :rtype: The the most frequent value in x.
    """

    vc = x.value_counts()
    if len(vc) > 0:
        index_names = vc.index.names
        vc = pd.DataFrame(vc)
        vc.columns = ['counts']
        vc = vc.reset_index()
        # sort to keep consistent output
        vc = vc.sort_values(['counts', 'index'], ascending=[False, True])
        vc = vc.set_index(['index'])
        vc.index.names = index_names
        return vc.index[0]
    else:
        return None


class GroupByImputer(BaseEstimator, TransformerMixin):
    """Imputes Missing Values by Group with specified function. If a ``group``
    parameter is given, it can be the name of any function which can be passed
    to the ``agg`` function of a pandas ``GroupBy`` object.  If a ``group``
    paramter is not given, then only 'mean', 'median', and 'most_frequent'
    can be used.


    :param impute_type:
        The type of imputation to be performed.
    :type impute_type: string
    :param group:
        The column name or a list of column names to group the ``pandas
        DataFrame``.
    :type group: string or list of strings
    """

    def __init__(self, impute_type, group=None):
        self.group = group
        if impute_type == 'most_frequent':
            self.impute_type = mode
        else:
            self.impute_type = impute_type

    def fit(self, X, y=None):
        """Fit the imputer on X

        :param X: The input data.
        :type X: pandas DataFrame
        :rtype: Returns self.
        """
        if self.group:
            self.mapper = X.groupby(self.group).agg(self.impute_type).to_dict()
        elif self.impute_type == mode:
            self.mapper = X.mode().iloc[0, :].to_dict()
        else:
            if self.impute_type == 'median':
                self.mapper = X.median().to_dict()
            elif self.impute_type == 'mean':
                self.mapper = X.mean().to_dict()
            else:
                raise ValueError(("Can only use 'most_frequent', 'median',"
                                  "or 'mean' impute_types without 'group'"
                                  "specified."))
        return self

    def _get_value_from_map(self, x, col):
        """get a value from the mapper, for a given column and a ``pandas Series``
        representing a row of data.

        :param x: A row of data from a ``DataFrame``.
        :type x: pandas Series
        :param col: The name of the column to impute a missing value.
        :type col: string
        :rtype:
            The value from self.mapper dictionary if exists, np.nan otherwise.
        """
        try:
            key = x[self.group]
            if isinstance(key, pd.Series):
                key = tuple(key)
            return self.mapper[col][key]
        except KeyError:
            return np.nan

    def transform(self, X):
        """Impute the eligible missing values in X

        :param X: The input data with missing values to be imputed.
        :type X: pandas DataFrame
        :rtype: A ``DataFrame`` with eligible missing values imputed.
        """
        X = X.copy()
        if self.group:
            for col in self.mapper.keys():
                X[col] = X[col].fillna(X.apply(
                    lambda x: self._get_value_from_map(x, col), axis=1))
        else:
            X = X.fillna(pd.Series(self.mapper))
        return X


class MissingValueFiller(BaseEstimator, TransformerMixin):
    """Fill missing values with a specified value.  Should only be used with
    columns of similar dtypes.

    :param value: The value to impute for missing factors.
    """

    def __init__(self, value):
        self.value = value

    def fit(self, X, y=None):
        """Fit the imputer on X.

        :param X: The input data.
        :type X: pandas DataFrame
        :rtype: Returns self.
        """
        return self

    def transform(self, X):
        """Impute the eligible missing values in X.

        :param X: The input data with missing values to be filled.
        :type X: pandas DataFrame
        :rtype: A ``DataFrame`` with eligible missing values filled.
        """
        X = X.fillna(self.value)
        return X


class OverMissingThresholdDropper(BaseEstimator, TransformerMixin):
    """Drop columns with more missing data than a given threshold.

    :param threshold: Maximum portion of missing data that is acceptable.  Must
                      be within the interval [0,1]
    :type threshold: float
    """

    def __init__(self, threshold):
        if threshold > 1 or threshold < 0:
            raise ValueError("threshold must be within [0,1]")
        else:
            self.threshold = threshold

    def fit(self, X, y=None):
        """Fit the dropper on X.

        :param X: The input data.
        :type X: pandas DataFrame
        :rtype: Returns self.
        """
        length = len(X)
        na_counts = X.isnull().sum()
        self.cols_to_drop = na_counts[
            (na_counts > int(length*(self.threshold)))].index.tolist()
        return self

    def transform(self, X):
        """Impute the eligible missing values in X.

        :param X: The input data.
        :type X: pandas DataFrame
        :rtype: A ``DataFrame`` with columns dropped.
        """
        X = X.drop(self.cols_to_drop, axis=1)
        return X


class ValueReplacer(BaseEstimator, TransformerMixin):
    """Replaces Values in each column according to a nested dictionary.
    ``inverse_mapper`` is probably more intuitive for when one value replaces
    many values.  Only one of ``inverse_mapper`` or ``mapper`` can be used.

    :param mapper: Nested dictionary with columns mapping to dictionaries
                   that map old values to new values.
    :type mapper: dictionary
    :param inverse_mapper: Nested dictionary with columns mapping to
                           dictionaries that map new values to a list of old
                           values
    :type inverse_mapper: dictionary

    ``mapper`` takes the form::

       {'column_name': {'old_value1': 'new_value1',
                        'old_value2': 'new_value1',
                        'old_value3': 'new_value2'}
        }

    while ``inverse_mapper`` takes the form::

       {'column_name': {'new_value1': ['old_value1', 'old_value2'],
                        'new_value2': ['old_value1']}
        }
    """

    def __init__(self, mapper=None, inverse_mapper=None):
        if inverse_mapper and mapper:
            raise ValueError("Cannot use both a mapper and inverse_mapper.")
        elif inverse_mapper:
            mapper = {}
            for k, d in inverse_mapper.items():
                map2 = {}
                for key, value in d.items():
                    for string in value:
                        map2[string] = key
                mapper[k] = map2
        elif not mapper:
            raise ValueError("Must initialize with either mapper or "
                             "inverse_mapper.")
        mapper = {key: dict_default(value) for key, value in mapper.items()}
        self.mapper = mapper

    def fit(self, X, y=None):
        """Fit the value replacer on X.  Checks that all columns in mapper are
        in present in X.

        :param X: The input data.
        :type X: pandas DataFrame
        :rtype: Returns self.
        """
        if len(set(self.mapper.keys()) - set(X.columns)) > 0:
            raise ValueError("Mapper contains columns not found in input"
                             "data: " +
                             ', '.join(set(self.mapper.keys())
                                       - set(X.columns)))
        return self

    def transform(self, X):
        """Replace the values in X with the values in the mapper.  Values not
        accounted for in the mapper will be left untransformed.

        :param X: The input data.
        :type X: pandas DataFrame
        :rtype: A ``DataFrame`` with old values mapped to new values.
        """
        X = X.copy(deep=True)
        for col in self.mapper.keys():
            X[col] = X[col].map(self.mapper[col])
        return X


class FactorLimiter(BaseEstimator, TransformerMixin):
    """For each named column, it limits factors to a list of acceptable values.
    Non-comforming factors, including missing values, are replaced by a default
    value.

    :param factors_per_column: dictionary mapping column name keys to a
                               dictionary with a list of acceptable factor
                               values and a default factor value for
                               non-conforming values
    :type factors_per_column: dictionary

    ``factors_per_column`` takes the form::

       {'column_name': {'factors': ['value1', 'value2', 'value3'],
                        'default': 'value1'},
                        }
        }
    """

    def __init__(self, factors_per_column=None):
        mapper = {}
        for col, specs in factors_per_column.items():
            # new_dict = dict_factory('new_dict', specs['default'])
            translation = {factor: factor for factor in specs['factors']}
            new_dict = dict_with_default(specs['default'], translation)
            # mapper[col] = new_dict(translation)
            mapper[col] = new_dict
        self.mapper = mapper

    def fit(self, X, y=None):
        """Fit the factor limiter on X.  Checks that all columns in
        factors_per_column are in present in X.

        :param X: The input data.
        :type X: pandas DataFrame
        :rtype: Returns self.
        """
        if len(set(self.mapper.keys()) - set(X.columns)) > 0:
            raise ValueError("factors_per_column contains keys not found in "
                             "DataFrame columns:" ', '.join(
                                 set(self.mapper.keys()) - set(X.columns)))
        return self

    def transform(self, X):
        """Limit the factors in X with the values in the factor_per_column.

        :param X: The input data.
        :type X: pandas DataFrame
        :rtype: A ``DataFrame`` with factors limited to the specifications.
        """
        X = X.copy(deep=True)
        for col, val in self.mapper.items():
            X[col] = X[col].map(val)
        return X


class SingleValueAboveThresholdDropper(BaseEstimator, TransformerMixin):
    """Removes columns with a single value representing a higher percentage
    of values than a given threshold

    :param threshold: percentage of single value in a column to be removed
    :type threshold: float
    :param dropna: If True, do not consider NaN as a value
    :type dropna: boolean
    """

    def __init__(self, threshold=1, dropna=True):
        if threshold > 1 or threshold < 0:
            raise ValueError("threshold must be within [0,1]")
        else:
            self.threshold = threshold
        self.dropna = dropna

    def fit(self, X, y=None):
        """Fit the dropper on X.

        :param X: The input data.
        :type X: pandas DataFrame
        :rtype: Returns self.
        """
        length = len(X)
        val_counts = X.apply(lambda x:
                             x.value_counts(dropna=self.dropna).iloc[0])
        self.cols_to_drop = val_counts[
            (val_counts >= int(length*(self.threshold)))].index.tolist()
        return self

    def transform(self, X):
        """Drop the columns in X with single values that exceed the threshold.

        :param X: The input data.
        :type X: pandas DataFrame
        :rtype: A ``DataFrame`` with columns dropped to the specifications.
        """
        X = X.drop(self.cols_to_drop, axis=1)
        return X


class SingleValueDropper(BaseEstimator, TransformerMixin):
    """Drop columns with only one unique value

    :param dropna: If True, do not consider NaN as a value
    :type dropna: boolean
    """

    def __init__(self, dropna=True):
        self.dropna = dropna

    def _unique_values(self, x):
        values = x.unique().tolist()
        if self.dropna and x.isnull().sum() > 0:
            if None in values:
                values.remove(None)
            values = [value for value in values if value == value]
        return len(values)

    def fit(self, X, y=None):
        """Fit the dropper on X.

        :param X: The input data.
        :type X: pandas DataFrame
        :rtype: Returns self.
        """
        val_counts = X.apply(self._unique_values, axis=0)
        self.cols_to_drop = val_counts[(val_counts <= 1)].index.tolist()
        return self

    def transform(self, X):
        """Drop the columns in X with single non-missing values.

        :param X: The input data.
        :type X: pandas DataFrame
        :rtype: A ``DataFrame`` with columns dropper.
        """
        X = X.drop(self.cols_to_drop, axis=1)
        return X


class ColumnExtractor(BaseEstimator, TransformerMixin):
    """Extract a list of columns from a ``DataFrame``.

    :param col: A list of columns to extract from the ``DataFrame``
    :type col: list of strings
    """
    def __init__(self, col):
        self.col = col

    def fit(self, X, y=None, **fit_params):
        """Fit the extractor on X. Checks that all columns are in X.

        :param X: The input data.
        :type X: pandas DataFrame
        :rtype: Returns self.
        """
        if len(set(self.col) - set(X.columns)) > 0:
            raise ValueError("Column list contains columns not found in input"
                             "data: " +
                             ', '.join(set(self.col) - set(X.columns)))
        return self

    def transform(self, X, **transform_params):
        """Extract the specified columns in X.

        :param X: The input data.
        :type X: pandas DataFrame
        :rtype: A ``DataFrame`` with specified columns.
        """
        return pd.DataFrame(X[self.col])


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Drop a list of columns from a ``DataFrame``.

    :param col: A list of columns to extract from the ``DataFrame``
    :type col: list of strings
    """
    def __init__(self, col):
        self.col = col

    def fit(self, X, y=None, **fit_params):
        """Fit the dropper on X. Checks that all columns are in X.

        :param X: The input data.
        :type X: pandas DataFrame
        :rtype: Returns self.
        """
        if len(set(self.col) - set(X.columns)) > 0:
            raise ValueError("Column list contains columns not found in input "
                             "data: " + ', '.join(set(self.col)
                                                  - set(X.columns)))
        return self

    def transform(self, X, **transform_params):
        """Drop the specified columns in X.

        :param X: The input data.
        :type X: pandas DataFrame
        :rtype: A ``DataFrame`` without specified columns.
        """
        return X.drop(self.col, axis=1)


class DummyCreator(BaseEstimator, TransformerMixin):
    """Create dummy variables from categorical variables.

    :param dummy_na: Add a column to indicate NaNs, if False NaNs are ignored.
    :type dummy_na: boolean
    :param drop_first: Whether to get k-1 dummies out of k categorical levels
                       by removing the first level.
    :type drop_first: boolean
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def _get_dummies(self, X, fit):
        if fit:
            return pd.get_dummies(X, **self.kwargs)
        else:
            new_kwargs = self.kwargs.copy()
            if 'drop_first' in self.kwargs:
                del new_kwargs['drop_first']
            return pd.get_dummies(X, **new_kwargs)

    def _fit(self, X):
        X = self._get_dummies(X, fit=True)
        self.columns = X.columns
        return X

    def fit(self, X, y=None, **fit_params):
        """Fit the dummy creator on X. Retains a record of columns produced
        with the fitting data.

        :param X: The input data.
        :type X: pandas DataFrame
        :rtype: Returns self.
        """
        self._fit(X)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Fit the dummy creator on X, then transform X.  Same as calling
        self.fit().transform(), but more convenient and efficient.

        :param X: The input data.
        :type X: pandas DataFrame
        :rtype: Returns self.
        """
        X = self._fit(X)
        return X

    def transform(self, X, **transform_params):
        """Create dummies for the columns in X.

        :param X: The input data.
        :type X: pandas DataFrame
        :rtype: A ``DataFrame`` with dummy variables.
        """
        X = self._get_dummies(X, fit=False)
        column_set = set(self.columns)
        data_column_set = set(X.columns)
        if column_set != data_column_set:
            # use same column order
            for col in self.columns:
                if col not in data_column_set:
                    X[col] = 0
        X = X[self.columns]
        return X


class ColumnValidator(BaseEstimator, TransformerMixin):
    """Ensure that the transformed dataset has the same columns and order as
    the original fit dataset. Could be useful to check at the beginning and
    end of pipelines.

    """

    def fit(self, X, y=None, **fit_params):
        """Fit the validator on X.

        :param X: The input data.
        :type X: pandas DataFrame
        :rtype: Returns self.
        """
        self.columns = X.columns
        return self

    def transform(self, X, **transform_params):
        """Checks whether a dataset to transform has the same columns as the
        fitting dataset, and returns X with columns in the same order as the
        dataset in fit.

        :param X: The input data.
        :type X: pandas DataFrame
        :rtype: A ``DataFrame`` with specified columns.
        """
        if len(set(self.columns) - set(X.columns)) > 0:
            raise ValueError("New data is missing columns from original data: "
                             + ', '.join(set(self.columns) - set(X.columns)))
        elif len(set(X.columns) - set(self.columns)) > 0:
            raise ValueError("New data has columns not in the original data: "
                             + ', '.join(set(X.columns) - set(self.columns)))
        return pd.DataFrame(X[self.columns], index=X.index)


class TextContainsDummyExtractor(BaseEstimator, TransformerMixin):
    """Extract one or more dummy variables based on whether one or more text
    columns contains one or more strings.

    :param mapper: a mapping of new columns to criteria to populate it as True
    :type mapper: dict

    ``mapper`` takes the form::

      {'old_column1':
       {'new_column1':
        [{'pattern': 'string1', 'kwargs': {'case': False}},
         {'pattern': 'string2', 'kwargs': {'case': False}}
         ],
        'new_column2':
        [{'pattern': 'string3', 'kwargs': {'case': False}},
         {'pattern': 'string4', 'kwargs': {'case': False}}
         ],
        },
       'old_column2':
       {'new_column3':
        [{'pattern': 'string5', 'kwargs': {'case': False}},
         {'pattern': 'string6', 'kwargs': {'case': False}}
         ],
        'new_column4':
        [{'pattern': 'string7', 'kwargs': {'case': False}},
         {'pattern': 'string8', 'kwargs': {'case': False}}
         ]
        }
       }
    """

    def __init__(self, mapper):
        self.mapper = mapper

    def fit(self, X, y=None):
        """Fit the imputer on X.

        :param X: The input data.
        :type X: pandas DataFrame
        :rtype: Returns self.
        """
        if len(set(self.mapper.keys()) - set(X.columns)) > 0:
            raise ValueError("Mapper contains columns not found in input"
                             "data: " +
                             ', '.join(set(self.mapper.keys())
                                       - set(X.columns)))
        return self

    def transform(self, X):
        """Impute the eligible missing values in X.

        :param X: The input data with missing values to be filled.
        :type X: pandas DataFrame
        :rtype: A ``DataFrame`` with eligible missing values filled.
        """
        X = X.copy(deep=True)
        for old_col, val in self.mapper.items():
            for new_col, terms in val.items():
                series_list = []
                for term in terms:
                    series_list.append(
                        X[old_col].str.contains(term['pattern'],
                                                **term['kwargs'])
                    )
                X[new_col] = bitwise_operator(
                    pd.DataFrame(series_list).transpose(), 'or').astype(int)
        return X


class BitwiseOperator(BaseEstimator, TransformerMixin):
    """Apply a bitwise operator ``&`` or ``|`` to a list of columns.

    :param mapper: A mapping from new columns which will be defined by applying
                   the bitwise operator to a list of old columns
    :type mapper: dict
    :param operator: the name of the bitwise operator to apply.
                     'and', 'or' are acceptable inputs
    :type operator: str

    ``mapper`` takes the form::

      {'new_column1': ['old_column1', 'old_column2', 'old_column3'],
       'new_column2': ['old_column2', 'old_column4', 'old_column5']
       }

    """

    def __init__(self, operator, mapper):
        self.mapper = mapper
        if operator in ['and', 'or']:
            self.operator = operator
        else:
            raise ValueError("parameter operator can only be 'and' or 'or'")

    def fit(self, X, y=None, **fit_params):
        """Fit the dropper on X. Checks that all columns are in X.

        :param X: The input data.
        :type X: pandas DataFrame
        :rtype: Returns self.
        """
        columns = [item for sublist in
                   [val for val in self.mapper.values()] for item in sublist]
        if len(set(columns) - set(X.columns)) > 0:
            raise ValueError("Column list contains columns not found in input "
                             "data:" + ', '.join(set(columns)
                                                 - set(X.columns)))
        return self

    def transform(self, X, **transform_params):
        """Drop the specified columns in X.

        :param X: The input data.
        :type X: pandas DataFrame
        :rtype: A ``DataFrame`` without specified columns.
        """
        X = X.copy(deep=True)
        for new_col, cols in self.mapper.items():
            X[new_col] = bitwise_operator(X[cols], self.operator).astype(int)
        return X


class BoxCoxTransformer(BaseEstimator, TransformerMixin):
    """Create BoxCox Transformations on all columns.

    :param adder: the amount to add to each column before the BoxCox
                  transformation
    :type adder: numeric
    """

    def __init__(self, adder=0):
        self.adder = adder

    def fit(self, X, y=None, **fit_params):
        """Fit the transformer on X.

        :param X: The input data.
        :type X: pandas DataFrame
        :rtype: Returns self.
        """
        self.columns = X.columns
        self.lambdas = dict()
        for col in self.columns:
            self.lambdas[col] = stats.boxcox(X[col] + self.adder)[1]
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Fit the validator on X.

        :param X: The input data.
        :type X: pandas DataFrame
        :rtype: Returns self.
        """
        X = X.copy()
        self.columns = X.columns
        self.lambdas = dict()
        for col in self.columns:
            X[col], self.lambdas[col] = stats.boxcox(X[col] + self.adder)
        return X

    def transform(self, X, **transform_params):
        """Checks whether a dataset to transform has the same columns as the
        fitting dataset, and returns X with columns in the same order as the
        dataset in fit.

        :param X: The input data.
        :type X: pandas DataFrame
        :rtype: A ``DataFrame`` with specified columns.
        """
        for col in self.lambdas:
            X[col] = stats.boxcox(X[col] + self.adder, self.lambdas[col])
        return X


class InteractionCreator(BaseEstimator, TransformerMixin):
    """Creates interactions across columns of a ``DataFrame``

    :param columns1: first list of columns to create interactions with each of
                     the second list of columns
    :type columns1: list of strings
    :param columns2: second list of columns to create interactions with each of
                     the second list of columns
    :type columns2: list of strings
    """
    def __init__(self, columns1, columns2):
        self.columns1 = columns1
        self.columns2 = columns2

    def fit(self, X, y=None, **fit_params):
        """Fit the transformer on X. Checks that all columns are in X.

        :param X: The input data.
        :type X: pandas DataFrame
        :rtype: Returns self.
        """
        if len((set(self.columns1) | set(self.columns2)) - set(X.columns)) > 0:
            raise ValueError("Column lists contains columns not found in input"
                             " data: " + ', '.join((set(self.columns1)
                                                    | set(self.columns2))
                                                   - set(X.columns)))
        formula = '0'
        for col1 in self.columns1:
            for col2 in self.columns2:
                formula = formula + '+' + col1 + ':' + col2
        self.formula = formula
        return self

    def transform(self, X, **transform_params):
        """Add specified interactions to X.

        :param X: The input data.
        :type X: pandas DataFrame
        :rtype: A ``DataFrame`` without specified columns.
        """

        model_matrix = dmatrix(self.formula, data=X, return_type='dataframe')
        return pd.concat([X, model_matrix], axis=1)


class StandardScaler(ScikitStandardScaler):
    """Standardize features by removing mean and scaling to unit variance
    """

    def __init__(self, columns=None, **kwargs):
        self.columns = columns
        super().__init__(**kwargs)

    def fit(self, X, y=None,  **fit_params):
        """Fit the transformer on X.

        :param X: The input data.
        :type X: pandas DataFrame
        :rtype: Returns self.
        """
        # assign columns if not defined at init
        if self.columns is None:
            self.columns = X.columns

        super().fit(X[self.columns])
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Fit and transform the StandardScaler on X.

        :param X: The input data.
        :type X: pandas DataFrame
        :rtype: Returns self.
        """
        X = X.copy()
        # assign columns if not defined at init
        if self.columns is None:
            self.columns = X.columns

        super().fit(X[self.columns])

        # transform proper columns
        X_transform = super().transform(X[self.columns])
        X_transform = pd.DataFrame(
            X_transform, columns=self.columns, index=X.index
        )

        # keep track of order and combine transform/non-transform columns
        cols_to_return = X.columns
        non_transformed_cols = [
            col for col in cols_to_return if col not in X_transform.columns
        ]
        X = pd.concat([X_transform, X[non_transformed_cols]], axis=1)

        # put columns back into original order
        X = X[cols_to_return]

        return X

    def transform(self, X, partial_cols=None, **transform_params):
        """Transform X with the standard scaling

        :param X: The input data.
        :type X: pandas DataFrame
        :param partial_cols: when specified, only return these columns
        :type X: list
        :rtype: A ``DataFrame`` with specified columns.
        """
        X = X.copy()
        # insert dummy columns into df if not provided
        if partial_cols is not None:
            for col in self.columns:
                if col not in X.columns:
                    X[col] = 0

        # remember order of original df
        cols_to_return = X.columns

        # transform columns in self.columns
        X_transform = super().transform(X[self.columns])
        X_transform = pd.DataFrame(
            X_transform, columns=self.columns, index=X.index
        )

        # add columns that weren't defined to be transformed back in
        non_transformed_cols = [
            col for col in cols_to_return if col not in X_transform.columns
        ]
        X = pd.concat([X_transform, X[non_transformed_cols]], axis=1)

        # put columns back into original order
        X = X[cols_to_return]

        # return only specified columns
        if partial_cols is not None:
            X = X[partial_cols]

        return X

    def inverse_transform(self, X, partial_cols=None, **transform_params):
        """Inverse transform X with the standard scaling

        :param X: The input data.
        :type X: pandas DataFrame
        :param partial_cols: when specified, only return these columns
        :type X: list
        :rtype: A ``DataFrame`` with specified columns.
        """
        X = X.copy()
        # insert dummy columns into df if not provided
        if partial_cols is not None:
            for col in self.columns:
                if col not in X.columns:
                    X[col] = 0

        # remember order of original df
        cols_to_return = X.columns

        # transform columns in self.columns
        X_transform = super().inverse_transform(
            X[self.columns]
        )
        X_transform = pd.DataFrame(
            X_transform, columns=self.columns, index=X.index
        )

        # add columns that weren't defined to be transformed back in
        non_transformed_cols = [
            col for col in cols_to_return if col not in X_transform.columns
        ]
        X = pd.concat([X_transform, X[non_transformed_cols]], axis=1)

        # put columns back into original order
        X = X[cols_to_return]

        # return only specified columns
        if partial_cols is not None:
            X = X[partial_cols]

        return X


class ColumnNameCleaner(BaseEstimator, TransformerMixin):
    """Replaces spaces and formula symbols in column names that conflict with
    patsy formula interpretation
    """

    def fit(self, X, y=None, **fit_params):
        """Fit the transformer on X.

        :param X: The input data.
        :type X: pandas DataFrame
        :rtype: Returns self.
        """
        matcher = re.compile(r'[^A-Z0-9_]', flags=re.IGNORECASE)
        self.columns = (X.columns
                        .str.strip()
                        .str.replace('+', '_and_')
                        .str.replace('*', '_by_')
                        .str.replace('/', '_or_')
                        .str.replace(matcher, '_'))
        print(self.columns)
        return self

    def transform(self, X, **transform_params):
        """Transform X with clean column names for patsy

        :param X: The input data.
        :type X: pandas DataFrame
        :rtype: A ``DataFrame`` with specified columns.
        """
        # ensure that columns are in same order as in fit
        X = X.copy()
        X.columns = self.columns
        return X


class PolynomialFeatures(BaseEstimator, TransformerMixin):
    """Creates polynomail features from inputs.

    :param degree: The degree of the polynomial
    :interaction_only: if true, only interaction features are produced:
    features that are products of at most degree distinct input features.

    """
    def __init__(self, degree=2, interaction_only=False):
        self.degree = degree
        self.interaction_only = interaction_only
        self.ScikitPolynomialFeatures = ScikitPolynomialFeatures(
            degree=self.degree,
            interaction_only=self.interaction_only,
            include_bias=False
        )

    def fit(self, X, y=None, **fit_params):
        """Fit the transformer on X.

        :param X: The input data.
        :type X: pandas DataFrame
        :rtype: Returns self.
        """
        self.columns = X.columns
        self.ScikitPolynomialFeatures.fit(X.values)

        # get polynomial feature names
        self.poly_feat = [
            str(e) for e in self.ScikitPolynomialFeatures.get_feature_names()
            if 'x' in e
        ]

        # for each polynomial feature name (x0, x1, etc)
        # map to df column name
        self.name_dict = OrderedDict()
        for n in np.arange(0, self.ScikitPolynomialFeatures.n_input_features_):
            self.name_dict[self.poly_feat[n]] = [self.columns[n]]

        # reverse OrderedDict to avoid name issues
        # eg., x1 & x11 confusion in column_name_string.replace()
        self.name_dict = OrderedDict(reversed(list(self.name_dict.items())))

        return self

    def transform(self, X, **transform_params):
        """Transform X with clean column names for patsy

        :param X: The input data.
        :type X: pandas DataFrame
        :rtype: A ``DataFrame`` with specified columns.
        """
        X = X.copy()[self.columns]
        X_transform = self.ScikitPolynomialFeatures.transform(X.values)

        # replace poly_feat names (x0, x1, etc.)
        # with actual column names and cleanup
        new_cols = self.poly_feat.copy()
        for poly_feat in self.name_dict.keys():
            for i, col in enumerate(new_cols):
                new_cols[i] = (
                    new_cols[i]
                    .replace(' ', '*')
                    .replace(poly_feat, self.name_dict[poly_feat][0])
                )

        # return df with original names used
        X_transform = pd.DataFrame(
            X_transform,
            columns=new_cols,
            index=X.index
        )

        return X_transform


class ContinuousFeatureBinner(BaseEstimator, TransformerMixin):
    """Creates bins for continuous features

    :param field: the continuous field for which to create bins
    :type field: string

    :param bins: The criteria to bin by.
    :type bins: array-like

    :param right_inclusive: interval should be right-inclusive or not
    :type right_inclusive: bool
    """
    def __init__(self, field, bins, right_inclusive=True):
        self.field = field
        self.bins = bins
        self.right_inclusive = right_inclusive

    def transform(self, df):
        if self.field not in df.columns:
            raise ValueError('Field not found in dataframe.')

        # use pandas.cut() to create bins
        df[str(self.field) + str('_GRP')] = pd.cut(
            x=df[self.field],
            bins=self.bins,
            right=self.right_inclusive
        )

        # return labels as strings
        df[str(self.field) + str('_GRP')] = (
            df[str(self.field) + str('_GRP')].astype('str')
        )

        # label everything not in a bin as 'Other'
        df[str(self.field) + str('_GRP')] = (
            df[str(self.field) + str('_GRP')]
            .replace('nan', np.NaN)
            .fillna(value='Other')
        )

        return df

    def fit(self, df):
        return self


class TypeExtractor(BaseEstimator, TransformerMixin):
    """Returns dataframe with only specified field type

    :param type: desired type; either 'numeric' or 'categorical'
    :type type: string
    """
    def __init__(self, type):
        self.type = type

    def transform(self, df, **transform_params):
        df = df[self.selected_fields]
        return df

    def fit(self, df, **fit_params):
        if self.type == 'numeric':
            df = df.select_dtypes(include=[np.number])
            self.selected_fields = list(df.columns)

        elif self.type == 'categorical':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            cat_cols = [col for col in df.columns if col not in numeric_cols]
            df = df[cat_cols]
            self.selected_fields = cat_cols

        print('Selected fields: ' + str(self.selected_fields))
        return self


class GenericTransformer(BaseEstimator, TransformerMixin):
    """Generic transformer that applies user-defined function within
    pipeline framework. Arbitrary callable should only make transformations
    and does not store any fit() parameters. Lambda functions are not supported
    as they cannot be pickled.

    :param function: arbitrary function to use as a transformer
    :type function: callable
    :param params: dict with function parameter name as key and parameter value
                   as value
    :type params: dict
    """
    def __init__(self, function, params=None):
        self.function = function
        self.params = params

    def transform(self, X, **transform_params):
        if self.params is None:
            X_transform = self.function(X)
        else:
            X_transform = self.function(X, **self.params)

        return X_transform

    def fit(self, X, y=None, **fit_params):
        return self


class MissingColumnsReplacer(BaseEstimator, TransformerMixin):
    """Fill in missing columns to a DataFrame
    :param cols: The expected list of columns.
    :param value: The value to fill the new columns with by default
    """
    def __init__(self, cols, value):
        self.cols = cols
        self.value = value

    def fit(self, X, y=None):
        """Fit the imputer on X.
        :param X: The input data.
        :type X: pandas DataFrame
        :rtype: Returns self.
        """
        return self

    def transform(self, X):
        """Impute the eligible missing values in X.
        :param X: The input data with missing values to be filled.
        :type X: pandas DataFrame
        :rtype: A ``DataFrame`` with eligible missing values filled.
        """
        X = X.copy(deep=True)
        new_cols = sorted(list(set(self.cols) - set(X.columns)))
        for col in new_cols:
            X[col] = np.nan
        X.loc[:, new_cols] = X[new_cols].fillna(self.value)
        return X
