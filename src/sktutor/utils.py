# -*- coding: utf-8 -*-


def dict_factory(name, default):
    """Return a subclass of dict with a default value

    :param name: name of the subclass
    :type name: string
    :param default: the default value returned by the dict instead of KeyError
    """
    def __missing__(self, key):
        return default
    new_class = type(name, (dict,), {"__missing__": __missing__})
    return new_class


class dict_default(dict):
    """Dictionary subclass that returns the key instead of a KeyError
    """
    def __missing__(self, key):
        return key


def bitwise_or(frame):
    """Returns the result of applying the bitwise ``|`` operator to a list of
    series

    :param frame: data with colums to apply the bitwise or operator to
    :type frame: list
    """
    num_cols = frame.shape[1]
    if num_cols == 1:
        return frame.iloc[:, 0]
    else:
        series = frame.iloc[:, 0]
        for i in range(1, num_cols):
            series = series | frame.iloc[:, i]
    return series


def bitwise_and(frame):
    """Returns the result of applying the bitwise ``|`` operator to a list of
    series

    :param frame: data with colums to apply the bitwise or operator to
    :type frame: list
    """
    num_cols = frame.shape[1]
    if num_cols == 1:
        return frame.iloc[:, 0]
    else:
        series = frame.iloc[:, 0]
        for i in range(1, num_cols):
            series = series & frame.iloc[:, i]
    return series
