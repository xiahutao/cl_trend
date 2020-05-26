# coding=utf-8
'''
Created on 7.9, 2018

@author: fang.zhang
'''
from __future__ import division
from backtest_func import *

if __name__ == '__main__':
    symble_lst = ['btcusdt', 'ethusdt', 'eosbtc', 'htbtc', 'xrpbtc', 'ethbtc']
    time_lst = [15]
    df_lst = []
    for tm in time_lst:

        for symble in symble_lst:
            data = pd.read_csv('data/state_ocmup_1440l' + '.csv')
            data_ = data[(data['ave_r'] > 0.005) & (data['max_retrace'] < 0.4) & (data['ann_ret'] > 0.3) &
                         (data['sharp'] > 1.2) & (data['trade_times'] > 10)]
            if len(data_) > 0:
                if len(data_) < 100:
                    df_lst.append(data_)
                else:
                    df_lst.append(data_.sort_values(['sharp'], ascending=False).head(100))
    tm_df = pd.concat(df_lst)
    lst = []
    for method, group in tm_df.groupby(['symble']):
        if len(group) >= 1:
            lst.append(group.sort_values(['sharp'], ascending=False).head(6))
    df = pd.concat(lst)
    print(df)
    df.to_csv('data/1440.csv')
