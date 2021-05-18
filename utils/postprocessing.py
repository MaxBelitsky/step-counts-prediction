import numpy as np
import pandas as pd


def from_supervised(data):
    """
    Turns a supervised sequence back into a normal sequence.
    
    :param numpy array data: pandas DataFrame or list; input sequence
    :returns: unsupervised sequence
    :rtype: numpy array
    """

    out = data[0]
    for i in range(1, data.shape[0]):
        out = np.append(out, data[i][-1])
    return out


def hours_to_date(stpes_hours, dates_hours):
    """
    Converts steps/hour to steps/date.

    :param DataFrame stpes_hours: df with steps/hour
    :param DataFrame dates_hours: df with dates and hours from the initial data
    :returns: steps/date
    :rtype: DataFrame
    """
    steps = pd.concat([stpes_hours, dates_hours], axis=1)
    dates = steps['date'].unique()
    hours = np.arange(24)
    idx = pd.MultiIndex.from_product((dates, hours), names=['date', 'hour'])
    steps = steps.set_index(['date', 'hour']).reindex(idx, fill_value=0).reset_index()
    return steps.groupby(['date'])['value'].sum().reset_index(name='Steps').Steps
