import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae


def TargetPeriodIndex(df, start, end):
    start = [ "{}".format(n).zfill(2) for n in start ]
    end = [ "{}".format(n).zfill(2) for n in end ]
    try:
        frm = df.query("dteday == '{}-{}-{}' and hr == {}".format(*start))
    except:
        raise ValueError('''
        スタート時刻が不正です。
        ''')
    try:
        to = df.query("dteday == '{}-{}-{}' and hr == {}".format(*end))
    except:
        raise ValueError('''
        エンド時刻が不正です。
        ''')
        
    return df[frm.index[0]:to.index[0]].index


def setShiftRange(shift_range):
    shift_range = np.append( shift_range, shift_range[-1]+1 )
    shift_range = shift_range[shift_range!=0]
    
    return shift_range


def TimeShiftDataFrame(df_timeseries, *args, **kwargs):
    length = df_timeseries.shape[0]
    df = df_timeseries
    
    for col, shift_range in kwargs.items():
        series = df_timeseries[col]
        shift_range = setShiftRange(shift_range)
        data = np.empty( (length, len(shift_range)) )

        data_cols = []
        
        for i, shift in enumerate(shift_range):
            if shift > 0:
                data_cols.append( "{}_{}later".format(col,abs(shift)) )
                for j in range(length):
                    try:
                        data[j][i] = series[j+shift+1]
                    except:
                        data[j][i] = np.nan
            elif shift < 0:
                data_cols.append( "{}_{}before".format(col,abs(shift)) )
                for j in range(length):
                    try:
                        data[j][i] = series[j+shift+1]
                    except:
                        data[j][i] = np.nan
            elif shift == 0:
                pass
        
        df = pd.concat( [df, pd.DataFrame(data).set_axis(data_cols, axis=1).set_axis([i+1 for i in range(length)])], axis=1 )
    
    return df 
