# -*- coding: utf-8 -*-
from sklearn.pipeline import (FeatureUnion as SKFeatureUnion,
                              _fit_transform_one, _name_estimators,
                              _transform_one)
from sklearn.externals.joblib import Parallel, delayed
import sklearn
import pandas as pd
import numpy as np

if sklearn.__version__ < '0.19.0':
    _sklearn_version = 'old'
else:
    _sklearn_version = 'new'


class FeatureUnion(SKFeatureUnion):

    def fit_transform(self, X, y=None, **fit_params):
        self._validate_transformers()
        if _sklearn_version == 'old':
            result = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_transform_one)(trans, name, weight, X, y,
                                            **fit_params)
                for name, trans, weight in self._iter())
        else:
            result = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_transform_one)(trans, weight, X, y,
                                            **fit_params)
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
        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.
        Returns
        -------
        X_t : array-like or sparse matrix, shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers.
        """
        if _sklearn_version == 'old':
            Xs = Parallel(n_jobs=self.n_jobs)(
                delayed(_transform_one)(trans, name, weight, X)
                for name, trans, weight in self._iter())
        else:
            Xs = Parallel(n_jobs=self.n_jobs)(
                delayed(_transform_one)(trans, weight, X)
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
    Parameters
    ----------
    *transformers : list of estimators
    n_jobs : int, optional
        Number of jobs to run in parallel (default 1).
    Returns
    -------
    f : FeatureUnion
   """
    n_jobs = kwargs.pop('n_jobs', 1)
    if kwargs:
        # We do not currently support `transformer_weights` as we may want to
        # change its type spec in make_union
        raise TypeError('Unknown keyword arguments: "{}"'
                        .format(list(kwargs.keys())[0]))
    return FeatureUnion(_name_estimators(transformers), n_jobs=n_jobs)
