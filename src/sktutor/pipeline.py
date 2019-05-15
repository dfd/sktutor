# -*- coding: utf-8 -*-
from sklearn.pipeline import (FeatureUnion as SKFeatureUnion,
                              _fit_transform_one, _name_estimators,
                              _transform_one)
from sklearn.externals.joblib import Parallel, delayed
import sklearn
import pandas as pd
import numpy as np

if sklearn.__version__ < '0.20.0':
    _sklearn_version = 'old'
else:
    _sklearn_version = 'new'


class FeatureUnion(SKFeatureUnion):
    """ Perform a list of transformations in parallel and concat the results

    :param transformers: list of (string, transformer) tuples
    :param n_jobs: Number of jobs to run in parallel (default 1).
    """

    def fit_transform(self, X, y=None, **fit_params):
        self._validate_transformers()
        if _sklearn_version == 'old':
            result = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_transform_one)(
                    transformer=trans,
                    name=name,
                    weight=weight,
                    X=X,
                    y=y,
                    **fit_params
                )
                for name, trans, weight in self._iter())
        else:
            result = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_transform_one)(
                    transformer=trans,
                    weight=weight,
                    X=X,
                    y=y,
                    **fit_params
                )
                for name, trans, weight in self._iter())
        if not result:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        Xs = pd.concat(Xs, axis=1)
        return Xs

    def transform(self, X):
        """Transform X separately by each transformer, concatenate results.

        :param X: Input data to be transformed.
        :type X: iterable or array-like, depending on transformers
        :rtype: DataFrame with concatenated results of transformers.
        """

        if _sklearn_version == 'old':
            Xs = Parallel(n_jobs=self.n_jobs)(
                delayed(_transform_one)(
                    transforme=trans,
                    name=name,
                    weight=weight,
                    X=X
                )
                for name, trans, weight in self._iter())
        else:
            Xs = Parallel(n_jobs=self.n_jobs)(
                delayed(_transform_one)(
                    transformer=trans,
                    weight=weight,
                    X=X,
                    y=None
                )
                for name, trans, weight in self._iter())
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        Xs = pd.concat(Xs, axis=1)
        return Xs


def make_union(*transformers, **kwargs):
    """Construct a FeatureUnion from the given transformers.
    This is a shorthand for the FeatureUnion constructor; it does not require,
    and does not permit, naming the transformers. Instead, they will be given
    names automatically based on their types. It also does not allow weighting.

    :param transformers: list of estimators
    :param n_jobs: Number of jobs to run in parallel (default 1).
    :rtype: FeatureUnion
    """

    n_jobs = kwargs.pop('n_jobs', 1)
    if kwargs:
        # We do not currently support `transformer_weights` as we may want to
        # change its type spec in make_union
        raise TypeError('Unknown keyword arguments: "{}"'
                        .format(list(kwargs.keys())[0]))
    return FeatureUnion(_name_estimators(transformers), n_jobs=n_jobs)
