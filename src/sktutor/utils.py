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


def bitwise_or(series_list):
    if len(series_list) == 1:
        return series_list[0]
    else:
        for i in range(len(series_list) - 1):
            series = series_list[i] | series_list[i+1]
    return series
