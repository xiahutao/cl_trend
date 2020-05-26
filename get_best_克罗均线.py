# coding=utf-8
'''
Created on 7.9, 2018

@author: fang.zhang
'''
from __future__ import division
from backtest_func import *

if __name__ == '__main__':
    time_lst = [30]
    df_lst = []
    for tm in time_lst:
        data = pd.read_csv('data/state_ocmup_' + str(tm) + '.csv')
        data = data[(data['ave_r'] > 0.005) & (data['max_retrace'] < 0.5) & (data['ann_ret'] > 0.3) &
                     (data['sharp'] > 0.5) & (data['trade_times'] > 2) & (data['win_r'] > 0.25)]
        # if len(data_) > 0:
        #     if len(data_) < 100:
        #         df_lst.append(data_)
        #     else:
        #         df_lst.append(data_.sort_values(['sharp'], ascending=False).head(100))
    # tm_df = pd.concat(df_lst)
    lst = []
    for method, group in data.groupby(['period_s', 'period_l', 'k_s', 'k_l', 'art_N', 'art_n']):
        row = []
        row.append(method)
        row.append(len(group))
        row.append(group.sharp.mean())
        row.append(group.ann_ret.mean())
        row.append(group.trade_times.mean())
        row.append(group.ave_hold_days.mean())
        lst.append(row)
        # if len(group) >= 1:
        #     lst.append(group.sort_values(['sharp'], ascending=False).head(5))

    df = pd.DataFrame(lst, columns=['method', 'num', 'sharp', 'ann_ret', 'trade_times', 'ave_hold_days'])
    print(df)
    df.to_csv('data/crowe_' + str(tm) + '.csv')
