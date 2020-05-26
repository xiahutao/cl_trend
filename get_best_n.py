# coding=utf-8
'''
Created on 7.9, 2018
@author: fang.zhang
'''
from __future__ import division
from backtest_func import *

if __name__ == '__main__':
    symble_lst = ['btcusdt', 'ethusdt', 'eosbtc', 'eoseth', 'htbtc', 'hteth', 'xrpbtc', 'yeebtc', 'yeeeth', 'ethbtc', 'htusdt']
    time_lst = ['240']

    df_lst = []
    for tm in time_lst:
        data = pd.read_csv('data/neu_strategy_factor135_' + str(tm) + '.csv')
        print(data)
        data = data[(data['ave_r'] > -1) & (data['max_retrace'] < 0.5) & (data['ann_ret'] > 0.1) &
                     (data['sharp'] > 0.7) & (data['trade_times'] > 0)]
        print(data)
        # if len(data_) > 0:
        #     if len(data_) < 100:
        #         df_lst.append(data_)
        #     else:
        #         df_lst.append(data_.sort_values(['sharp'], ascending=False).head(100))
    # tm_df = pd.concat(df_lst)
    lst = []
    for method, group in data.groupby(['N1', 'N2', 'N']):
        # if len(group)==3:
        #     lst.append(group)
        row = []
        row.append(method)
        row.append(len(group))
        row.append(group.sharp.mean())
        row.append(group.ann_ret.mean())
        row.append(group.trade_times.mean())
        row.append(group.win_r.mean())
        row.append(group.odds.mean())
        row.append(group.max_retrace.mean())
        row.append(group.ave_r.mean())
        lst.append(row)
        # if len(group) >= 1:
        #     lst.append(group.sort_values(['sharp'], ascending=False).head(5))

    df = pd.DataFrame(lst, columns=['method', 'num', 'sharp', 'ann_ret', 'trade_times', 'win_r', 'odds', 'max_retrace',
                                    'ave_r'])
    print(df)
    df.to_csv('data/factor135_' + tm + '.csv')
    # pd.concat(lst).to_csv('alpha_usdt_5' + str(tm) + '.csv')
