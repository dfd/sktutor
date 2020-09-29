# -*- coding: utf-8 -*-
from sklearn.pipeline import (FeatureUnion as SKFeatureUnion,
                              _fit_transform_one, _name_estimators,
                              _transform_one)
from joblib import Parallel, delayed
import sklearn
import pandas as pd
import inspect
import numpy as np


class FeatureUnion(SKFeatureUnion):
    """ Perform a list of transformations in parallel and concat the results

    :param transformers: list of (string, transformer) tuples
    :param n_jobs: Number of jobs to run in parallel (default 1).
    """
    
    def fit_args(self, func, local,X=None,y=None):
        sig = inspect.signature(func)
        arg_dict = {}
        fit_params = {}
        for i in sig.parameters.values():
            if i.name == 'transformer':
                arg_dict[i.name]=local['trans']
            elif '**' in str(i):
                try:
                    fit_params = local[i.name]
                except:
                    pass
            else:
                try:
                    arg_dict[i.name]=local[i.name]
                except:
                    arg_dict[i.name]=None
        arg_dict['X']=X
        return arg_dict,fit_params
    
    def fit_transform(self, X, y=None, **fit_params):
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(
                **(self.fit_args(_fit_transform_one,locals(),X,y)[0]),
                **(self.fit_args(_fit_transform_one,locals())[1])
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
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(
                **(self.fit_args(_transform_one,locals(),X)[0]),
                **(self.fit_args(_transform_one,locals())[1])
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
