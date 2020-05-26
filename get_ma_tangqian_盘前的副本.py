# coding=utf-8
'''
Created on 8.8, 2017

@author: fang.zhang
'''
from __future__ import division
import numpy as np
import pandas as pd
import os
import talib
import time
import datetime
import copy
from mssqlDBConnectorPeng import *


if __name__ == '__main__':
    os.getcwd()
    os.chdir(r'F:\\Program Files\\PycharmProjects\\indicators')
    t0 = time.time()
    N1_lst = [60]
    ATR_n_lst = [2, 2.5]
    # N3_lst = [25, 35, 45, 60, 80, 120]
    P_ATR_LST = [11, 15, 20]
    road_period_lst = [20]
    win_stop_lst = [0.2]

    csd = MysqlDBConnector('db_production')
    stockdata = pd.read_hdf('data/data_2010.h5', 'all').reset_index(drop=False)
    print(stockdata)

    transfer_to_bool = lambda s: 1 if s else 0
    df_lst = []
    lst = []
    for N1 in N1_lst:
        for ATR_n in ATR_n_lst:
            for road_period in road_period_lst:
                for N_ATR in P_ATR_LST:
                    for win_stop in win_stop_lst:
                        print(N_ATR)
                        method = 'tqa' + str(N1) + '_' + str(N_ATR) + '_' + str(ATR_n) + '_' + str(road_period) + '_' + str(win_stop)
                        signal_lst = []
                        trad_times = 0
                        for code, group in stockdata.groupby(['stock_code']):
                            print(code)
                            if len(group) > N1:
                                group['h10'] = talib.MAX(group['high'].values, road_period)
                                group['l10'] = talib.MIN(group['low'].values, road_period)
                                group['ma'] = talib.MA(group['close'].values, N1)
                                group['atr'] = talib.ATR(group['high'].values, group['low'].values, group['close'].values, N_ATR)
                                group = group.assign(h10=lambda df: df.h10.shift(1))\
                                    .assign(l10=lambda df: df.l10.shift(1))\
                                    .assign(atr=lambda df: df.atr.shift(1))\
                                    .assign(c_ma=lambda df: df.close - df.ma)\
                                    .assign(ma_zd=lambda df: df.ma - df.ma.shift(1))
                                position = 0
                                #signal_row = []
                                #stock_row = []
                                for idx, _row in group.iterrows():
                                    if (position == 0) & (_row.high > _row.h10) & (_row.c_ma > 0) & (_row.ma_zd > 0):
                                        position = 1
                                        s_time = _row.date_time
                                        cost = _row.close
                                        hold_price = []
                                        high_price = []
                                        hold_price.append(_row.close)
                                        high_price.append(cost)
                                    elif position == 1:
                                        if _row.low < max(hold_price) - _row.atr * ATR_n:
                                            position = 0
                                            trad_times += 1
                                            high_price.append(_row.high)
                                            e_time = _row.date_time
                                            s_price = max(hold_price) - _row.atr * ATR_n
                                            ret = s_price / cost - 1
                                            signal_row = []
                                            signal_row.append(s_time)
                                            signal_row.append(e_time)
                                            signal_row.append(code)
                                            signal_row.append(cost)
                                            signal_row.append(s_price)
                                            signal_row.append(_row.close)
                                            signal_row.append(ret)
                                            signal_row.append(max(high_price)/cost - 1)
                                            signal_row.append(len(hold_price))
                                            signal_lst.append(signal_row)
                                        elif _row.high > cost * (1 + win_stop):
                                            position = 0
                                            trad_times += 1
                                            high_price.append(_row.high)
                                            e_time = _row.date_time
                                            s_price = cost * (1 + win_stop)
                                            ret = win_stop
                                            signal_row = []
                                            signal_row.append(s_time)
                                            signal_row.append(e_time)
                                            signal_row.append(code)
                                            signal_row.append(cost)
                                            signal_row.append(s_price)
                                            signal_row.append(_row.close)
                                            signal_row.append(ret)
                                            signal_row.append(max(high_price)/cost - 1)
                                            signal_row.append(len(hold_price))
                                            signal_lst.append(signal_row)

                                        else:
                                            high_price.append(_row.high)
                                            hold_price.append(_row.close)
                    signal_state = pd.DataFrame(signal_lst,
                                                columns=['s_time', 'e_time', 'stock_code', 'cost', 's_price', 'close', 'ret', 'max_ret', 'hold_day'])\
                        .assign(method=method)
                    signal_state.to_hdf('data/signal_tqa.h5', method)
                    csd.write_data_to_db(signal_state, 'zf_indicator_signal_tangqian', mode=3)


