from typing import List

import numpy as np
import pandas as pd

import datetime

def dayofyear(index):
    year = index.year
    month = index.month
    day = index.day
    date1 = datetime.date(year=int(year),month=int(month),day=int(day))
    date2 = datetime.date(year=int(year),month=1,day=1)
    return (date1 - date2).days + 1

def DayOfYear(index):
    """Day of year encoded as value between [-0.5, 0.5]"""
    return (dayofyear(index) - 1) / 365.0 - 0.5


def MonthOfYear(index):
    """Month of year encoded as value between [-0.5, 0.5]"""
    return (index.month - 1) / 11.0 - 0.5

def Year(index):
    """Month of year encoded as value between [-0.5, 0.5]"""
    return (index.year - 1) / 2100.0 - 0.5


def time_features(dates):
    year=Year(dates)
    month=MonthOfYear(dates)
    day=DayOfYear(dates)
    return np.hstack([year,month,day])
