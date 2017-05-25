# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.pipeline import FeatureUnion
# from sklearn.pipeline import Pipeline
import numpy as np


def mode(x):
    """return the most frequent occurance.  If two or more values are tied
    with the most occurances, then return the lowest value

    :param x: a pandas Series
    """

    vc = x.value_counts()
    if len(vc) > 0:
        index_names = vc.index.names
        vc = pd.DataFrame(vc)
        vc.columns = ['counts']
        vc = vc.reset_index()
        print(vc.columns)
        # sort to keep consistent output
        vc = vc.sort_values(['counts', 'index'], ascending=[False, True])
        vc = vc.set_index(['index'])
        vc.index.names = index_names
        return vc.index[0]
    else:
        return None


class ImputeByGroup(BaseEstimator, TransformerMixin):
    """Imputes Missing Values by Group with specified function. If a `group`
    parameter is given, it can be the name of any function which can be passed
    to the `agg` function of a pandas `GroupBy` object.  If a `group` paramter
    is not given, then only 'mean', 'median', and 'most_frequent' can be used.

    :param impute_type:
        string representing the type of imputation to be performed. 
    :type impute_type: string
    :param group: the column or a list of columns to group the `DataFrame`
    :type group: string or list
    """

    def __init__(self, impute_type, group=None):
        self.group = group
        if impute_type == 'most_frequent':
            self.impute_type = mode
        else:
            self.impute_type = impute_type

    def fit(self, X, y=None):
        """fit the imputer on X

        :param X: a pandas `DataFrame`
        :rtype: returns self
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
        """get a value from the mapper, for a given column and a `pandas Series`
        representing a row of data.

        :param x: a `pandas Series`
        :param col: the name of the column to  impute a missing value
        :rtype: value from self.mapper dictionary if found, np.nan otherwise
        """
        try:
            return self.mapper[col][x[self.group]]
        except KeyError:
            return np.nan

    def transform(self, X):
        """Impute the eligible missing values in X

        :param X: a `DataFrame` with missing values to be imputed
        :rtype: a `DataFrame` with eligible missing values imputed
        """
        X = X.copy()
        if self.group:
            for col in self.mapper.keys():
                X[col] = X[col].fillna(X.apply(
                    lambda x: self._get_value_from_map(x, col), axis=1))
        else:
            X = X.fillna(pd.Series(self.mapper))
        return X
