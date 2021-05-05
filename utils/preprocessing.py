import numpy as np
import pandas as pd
import pytz
import datetime
import holidays


def to_supervised(sequence, n_previous, n_future):
    """
    
    :param numpy array sequence: pandas DataFrame or list; input sequence
    :param n_previous: int; number of past data points (N)
    :param n_future: int; number of data points to be predicted (K)
    :returns:
        - X: (seq. length x n_previous) numpy array with the previous observations
        - y: (seq. length x n_fututre) numpy array with the target observations
    """
    
    X, y = [], []
    idx, i = 0, 0
    while idx+n_future <= len(sequence)-1:
        idx = i + n_previous
        X.append(sequence[i:idx])
        y.append(sequence[idx:idx+n_future])
        i += 1
        
    return np.array(X), np.array(y)


def convert_steps(steps, exclude_corona=True):
    """
    Adds converts starts date of each entry to datetime object and
    adds year, month, date, day, hour, minute and day of the week to each entry

    :param DataFrame steps: the raw steps data
    :param bool exclude_corona: if True dicard the values after the beginning of coronavirus
    :return: converted steps
    :rtype: DataFrame
    """

    convert_tz = lambda x: x.to_pydatetime().replace(tzinfo=pytz.utc).astimezone(pytz.timezone('Europe/Vilnius'))
    get_year = lambda x: convert_tz(x).year
    get_month = lambda x: '{}-{:02}'.format(convert_tz(x).year, convert_tz(x).month) #inefficient
    get_date = lambda x: '{}-{:02}-{:02}'.format(convert_tz(x).year, convert_tz(x).month, convert_tz(x).day) #inefficient
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
    Aggregates the steps by date or by hour

    :param steps: pandas DataFrame; the raw steps data
    :param grouping: list; aggregate by date or by hour passing ['date', 'hour']
    :returns: pandas DataFrame; aggregated steps
    """
    return convert_steps(steps).groupby(grouping)['value'].sum().reset_index(name='Steps')


def augment(steps, add_month=True, add_year=True,
            add_weekend=True, add_holiday=True):
    """
    Augments the data with weekend and holiday data and
    converts the date column to sin and cos so that the models
    have the information about month and year periods

    :param steps: pandas DataFrame; the raw steps data
    :param add_month: bool; indicates whether to include periodicity by month
    :param add_year: bool; indicates whether to include periodicity by year
    :param add_weekend: bool; indicates whether to include weekend data
    :param add_holiday: bool; indicates whether to include holiday data
    :returns: pandas DataFrame; augmented steps
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

    # Remove the date column
    temp = temp.drop('date', axis=1)
    return temp
