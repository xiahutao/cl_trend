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


def dealdata_addcloselst(stockdata, count):
    timelistraw = list(stockdata['Date Time'])
    pricelistraw = np.float64(list(stockdata['close']))
    stockcoderaw = list(stockdata['STOCKCODE'])
    alllist = []
    for i in range(len(pricelistraw) - count + 1):
        row = []
        datalisttmp = []
        for j in range(count):
            datalisttmp.append(pricelistraw[i + j])
        row.append(datalisttmp)
        row.append(pricelistraw[i + count - 1])
        row.append(stockcoderaw[i + count - 1])
        row.append(timelistraw[i + count - 1])

        alllist.append(row)
    dealdatadf = pd.DataFrame(alllist, columns=['Datalist', 'close', 'STOCKCODE', 'Date Time'])
    return dealdatadf


def return_g_d(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


def normal_b_s(lst):
    b_s_lst = copy.deepcopy(lst)
    if lst[0] == -1:
        for i in range(len(lst)):
            if b_s_lst[i] > -1:
                b_s_lst[:i] = [0] * i
                b_s_lst[i] = 1
                break
    if lst[0] == 0:
        b_s_lst[0] = 1
    for j in range(1, len(b_s_lst)):
        if b_s_lst[j] == 0:
            b_s_lst[j] = -b_s_lst[j - 1]
    b_s_lst_ = copy.deepcopy(b_s_lst)
    for m in range(1, len(b_s_lst_)):
        if b_s_lst[m] * b_s_lst[m - 1] > 0:
            b_s_lst_[m] = 0
    for n in range(1, len(lst)):
        if b_s_lst_[-n] == 1:
            b_s_lst_[-n] = 0
            break
    return b_s_lst_


if __name__ == '__main__':
    os.getcwd()
    os.chdir(r'F:\\Program Files\\PycharmProjects\\indicators')
    t0 = time.time()
    x = 1
    s_time = '20100101'
    e_time = '20171116'
    all_lst = []
    stat_lst = []
    csd = MysqlDBConnector('db_production')

    stockdata = pd.read_hdf('data/data.h5', 'all')

    for x in [9]:
        try:
            all_lst.append(pd.read_hdf('data/signal_all_' + str(x) + '.h5', 'all'))
        except Exception as e:
            print(str(e))
    #all_lst.append(signal_all_1)
    #del signal_all_1
    signal_all = pd.concat(all_lst)\
        .sort_values(by=['method', 'stock_code', 'date_time'])
    print(signal_all)
    signal_all = signal_all.drop_duplicates(['method', 'stock_code', 'date_time'])\
        .sort_values(by=['method', 'stock_code', 'date_time'])
    print(signal_all)
    signal_all = signal_all[(signal_all['date_time'] > s_time) & (signal_all['date_time'] < e_time)]
    print(signal_all)
    lst = []
    signal_all_lst = []
    for method, group in signal_all.groupby(['method', 'stock_code']):
        # print(method)
        if group.iat[0, 3] == -1:
            group = group.iloc[1:, :]
        if len(group) == 1:
            print('lenth=1')
        if len(group) > 1:
            if len(group) % 2 != 0:
                print(len(group))
                group = group.iloc[:len(group)-1, :]
            signal_all_lst.append(group)
    signal_all = pd.concat(signal_all_lst)
    print(signal_all)
    signal_all[signal_all['stock_code']=='603345.sh'].to_csv('data/000538.csv')
    '''
    signal_all_1 = pd.read_hdf('data/signal_all.h5', 'all')
    print(signal_all_1)
    signal_all_1 = signal_all_1.append(signal_all)\
        .drop_duplicates(['method', 'stock_code', 'date_time']) \
        .sort_values(by=['method', 'stock_code', 'date_time'])
    print(signal_all_1)

    signal_all_1.to_hdf('data/signal_all_n.h5', 'all')
    del signal_all_1
    '''

    for method, group in signal_all.groupby(['method']):
        print(method)

        group = group.sort_values(by=['stock_code', 'date_time'])[[
            'date_time', 'stock_code', 'ret', 'b_s', 'method', 'close']]
        signal_method = group.assign(s_time=lambda df: df.date_time.shift(1))\
                            .assign(close_b=lambda df: df.close.shift(1))
        signal_method = signal_method\
                            .iloc[1::2, :][['date_time', 's_time', 'method', 'stock_code', 'ret', 'close', 'close_b']]
        group_ret_lst = signal_method.ret.tolist()
        win_lst = [i - 1 for i in group_ret_lst if i > 1]
        loss_lst = [i - 1 for i in group_ret_lst if i < 1]
        if (len(win_lst) > 0) & (len(loss_lst) > 0):
            win_R = len(win_lst) / (len(win_lst) + len(loss_lst))
            odds = -sum(win_lst) * len(loss_lst) / sum(loss_lst) / len(win_lst)
            ave_ret = (sum(win_lst) + sum(loss_lst)) / (len(win_lst) + len(loss_lst))
            ret_lst = []
            ret_lst_alpha = []
            max_ret_lst = []
            hold_day_lst = []
            for code, signal_one in signal_method.groupby(['stock_code']):
                print(code)
                ret_ori = signal_one.close.tolist()[-1]/signal_one.close_b.tolist()[0]
                signal_ret_lst = signal_one.ret.tolist()
                ret_cl = np.prod(signal_ret_lst)
                ret_lst_alpha.append(ret_cl - ret_ori)
                ret_lst.append(ret_ori)
                for idx, row in signal_one.iterrows():
                    hold_data = stockdata.loc[code, 'high'][row.s_time: row.date_time]
                    max_ret_lst.append(hold_data.max() / stockdata.loc[code, 'close'][row.s_time])
                    hold_day_lst.append(len(hold_data))
            signal_method = signal_method.assign(max_ret=max_ret_lst)\
                .assign(hold_day=hold_day_lst)
            try:
                csd.write_data_to_db(signal_method, 'zf_indicator_signal', mode=3)
            except Exception as e:
                print(str(e))
            win_lst_3 = [i - 1.03 for i in max_ret_lst if i > 1.03]
            loss_lst_3 = [i - 1.03 for i in max_ret_lst if i < 1.03]
            win_R_3 = len(win_lst_3) / (len(win_lst_3) + len(loss_lst_3))

            win_lst_5 = [i - 1.05 for i in max_ret_lst if i > 1.05]
            loss_lst_5 = [i - 1.05 for i in max_ret_lst if i < 1.05]
            win_R_5 = len(win_lst_5) / (len(win_lst_5) + len(loss_lst_5))

            win_lst_10 = [i - 1.10 for i in max_ret_lst if i > 1.10]
            loss_lst_10 = [i - 1.10 for i in max_ret_lst if i < 1.10]
            win_R_10 = len(win_lst_10) / (len(win_lst_10) + len(loss_lst_10))

            win_lst_code = [i - 1 for i in ret_lst if i > 1]
            loss_lst_code = [i - 1 for i in ret_lst if i < 1]

            win_lst_code_alpha = [i for i in ret_lst_alpha if i > 0]
            loss_lst_code_alpha = [i for i in ret_lst_alpha if i < 0]
            if (len(win_lst_code) > 0) & (len(loss_lst_code) > 0) & (len(win_lst_code_alpha) > 0) & (
                len(loss_lst_code_alpha) > 0):
                win_R_code = len(win_lst_code) / (len(win_lst_code) + len(loss_lst_code))
                win_R_code_alpha = len(win_lst_code_alpha) / (len(win_lst_code_alpha) + len(loss_lst_code_alpha))
                odds_code = -sum(win_lst_code) * len(loss_lst_code) / sum(loss_lst_code) / len(win_lst_code)
                odds_code_alpha = -sum(win_lst_code_alpha) * len(loss_lst_code_alpha) / sum(loss_lst_code_alpha) / len(
                    win_lst_code_alpha)
                row_ = []
                row_.append(method)
                row_.append(len(group_ret_lst))
                row_.append(ave_ret)
                row_.append(win_R)
                row_.append(odds)
                row_.append(win_R_code)
                row_.append(odds_code)
                row_.append(win_R_code_alpha)
                row_.append(odds_code_alpha)
                row_.append(win_R_3)
                row_.append(win_R_5)
                row_.append(win_R_10)
                row_.append(sum(hold_day_lst)/len(hold_day_lst))
                lst.append(row_)
    df = pd.DataFrame(lst, columns=['method', 'signal_count', 'ave_ret', 'win_R', 'odds', 'win_R_code', 'odds_code'
        , 'win_R_code_alpha', 'odds_code_alpha', 'win_R_3', 'win_R_5', 'win_R_10', 'ave_holddays'])\
        .assign(s_time=s_time)\
        .assign(e_time=e_time)
    csd.write_data_to_db(df, 'zf_indicator_a', mode=3)

    df.to_csv('data/method_stat_all' + s_time + '-' + e_time + '.csv')
