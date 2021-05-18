import numpy as np
import pandas as pd
import pytz
import datetime
import holidays


def to_supervised(sequence, n_previous, n_future):
    """
    Turns a sequence into a supervised sequence.
    
    :param numpy array sequence: input sequence
    :param int n_previous: number of past data points (N)
    :param int n_future: number of data points to be predicted (K)
    :returns:
        X: (seq. length x n_previous) numpy array with the previous observations
        y: (seq. length x n_fututre) numpy array with the target observations
    :rtypes: numpy arrays
    """
    
    X, y = [], []
    idx, i = 0, 0
    while idx+n_future <= len(sequence)-1:
        idx = i + n_previous
        X.append(sequence[i:idx])
        y.append(sequence[idx:idx+n_future])
        i += 1

    return np.array(X), np.array(y).squeeze()


def convert_steps(steps, exclude_corona=True):
    """
    Converts start date of each entry to datetime object and
    adds year, month, date, day, hour, minute and day of the week to each entry.
    Taken from https://github.com/markwk/qs_ledger/tree/master/apple_health

    :param DataFrame steps: the raw steps data
    :param bool exclude_corona: if True dicard the values after the beginning of pandemic
    :return: converted steps
    :rtype: DataFrame
    """

    convert_tz = lambda x: x.to_pydatetime().replace(tzinfo=pytz.utc).astimezone(pytz.timezone('Europe/Vilnius'))
    get_year = lambda x: convert_tz(x).year
    get_month = lambda x: '{}-{:02}'.format(convert_tz(x).year, convert_tz(x).month)
    get_date = lambda x: '{}-{:02}-{:02}'.format(convert_tz(x).year, convert_tz(x).month, convert_tz(x).day)
    get_day = lambda x: convert_tz(x).day
    get_hour = lambda x: convert_tz(x).hour
    get_minute = lambda x: convert_tz(x).minute
    get_day_of_week = lambda x: convert_tz(x).weekday()

    steps['startDate'] = pd.to_datetime(steps['startDate'])
    steps['year'] = steps['startDate'].map(get_year)
    steps['month'] = steps['startDate'].map(get_month)
    steps['date'] = steps['startDate'].map(get_date)
    steps['day'] = steps['startDate'].map(get_day)
    steps['hour'] = steps['startDate'].map(get_hour)
    steps['dow'] = steps['startDate'].map(get_day_of_week)

    if exclude_corona:
        corona_start = steps[steps.date == "2020-03-10"].index[0]
        steps = steps.iloc[:corona_start, :]

    return steps


def aggregate_steps(steps, grouping=['date']):
    """
    Aggregates the steps by date or by hour.

    :param DataFrame steps: the raw steps data
    :param list grouping: aggregate by date or by hour passing ['date', 'hour']
    :returns: aggregated steps
    :rtype: DataFrame
    """
    if grouping == ['date', 'hour']:
        return convert_steps(steps).groupby(['date', 'hour'], as_index=False).agg(
            {'value': 'sum', 'date': 'first', 'hour': 'first'}
        )

    return convert_steps(steps).groupby(grouping, as_index=False).agg(
        {'value': 'sum', 'date': 'first', 'dow': 'first'}
    )


def augment(steps, add_month=True, add_year=True,
            add_weekend=True, add_holiday=True):
    """
    Augments the data with weekend and holiday data and
    converts the date column to sin and cos so that the models
    have access to the information about month and year periods.

    :param DataFrame steps: the raw steps data
    :param bool add_month: indicates whether to include periodicity by month
    :param bool add_year: indicates whether to include periodicity by year
    :param bool add_weekend: indicates whether to include weekend data
    :param bool add_holiday: indicates whether to include holiday data
    :returns: augmented steps
    :rtype: DataFrame
    """

    temp = aggregate_steps(steps)
    temp.date = pd.to_datetime(temp.date, format='%Y-%m-%d')

    # Add holidays and weekends
    is_weekend = []
    is_holiday = []
    nl_holidays = holidays.Netherlands()
    for i in temp.date:
        if i.weekday() in [5, 6]:
            is_weekend.append(1)
        else:
            is_weekend.append(0)

        if i in nl_holidays:
            is_holiday.append(1)
        else:
            is_holiday.append(0)

    if add_weekend:
        temp['weekend'] = is_weekend
    if add_holiday:
        temp['holiday'] = is_holiday

    # Convert time to sin and cos to keep track of periodicity
    day = 24*60*60
    month = 30.416*day
    year = 365.2425*day

    timestamp_s = temp.date.map(datetime.datetime.timestamp)
    if add_month:
        temp['Month sin'] = np.sin(timestamp_s * (2 * np.pi / month))
        temp['Month cos'] = np.cos(timestamp_s * (2 * np.pi / month))
    if add_year:
        temp['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
        temp['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

    # Remove the date and day of the week columns
    temp = temp.drop('date', axis=1)
    temp = temp.drop('dow', axis=1)
    return temp
