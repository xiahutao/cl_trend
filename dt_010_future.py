# coding=utf-8
'''
dual trust策略
Created on 2018-11-06

@author: fang.zhang
'''
from __future__ import division

import datetime
import os
import time
import traceback

import pandas as pd
import talib

from restapi.realdataapi import get_exsymbol_kline

from utils.db_utils2 import upsert_longshort_signal_future
import sys
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'


def get_second_from_period(period):
    if period in ['15min', '30min', '60min']:
        return int(period[:2]) * 60
    elif period == '4hour':
        return 240 * 60
    else:
        return 1440 * 60


def get_road_from_period(period):
    if period == '1day':
        return 3, 3, 0.65, 0.45, 15
    elif period == '60min':
        return 38, 23, 0.4, 0.45, 10
    elif period == '4hour':
        return 7, 10, 0.7, 0.4, 15
    else:
        return 3, 20, 3, 20


def run_dt_future():

    stratid = 10
    now = datetime.datetime.now()
    start = (now - datetime.timedelta(days=25)).strftime('%Y-%m-%d')
    end = (now + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    # os.chdir(path)
    period_lst = ['60min', '4hour', '1day']
    coin_lst = ['.bxbt', '.beth']
    exchange = 'BITMEX'
    lst1 = []
    for period in period_lst:
        print(period)

        second_per_period = get_second_from_period(period)
        road_period_l, road_period_s, K1, K2, N1 = get_road_from_period(period)  # 获取通道周期、做多K值、做空K值、均线周期参数

        now = int(time.time())

        for name in coin_lst:
            try:
                print(name)
                errcode, errmsg, data_ = get_exsymbol_kline(exchange, name, period, start, end)
                data = data_.loc[:, ['tickid', 'date', 'high', 'low', 'close', 'open']]
                errcode, errmsg, day_data = get_exsymbol_kline(exchange, name, '1day', start, end)
                day_data = day_data.loc[:, ['tickid', 'date', 'high', 'low', 'close', 'open']] \
                    .head(len(day_data) - 1)\
                    .assign(ma=lambda df: talib.MA(df.close.values, N1)).reset_index()
                ma_1 = day_data.at[len(day_data) - 1, 'ma']
                ma_2 = day_data.at[len(day_data) - 2, 'ma']

                if now - second_per_period < data.tickid.tolist()[-1]:
                    data = data\
                        .assign(HH_l=lambda df: talib.MAX(df.high.values, road_period_l)) \
                        .assign(HC_l=lambda df: talib.MAX(df.close.values, road_period_l)) \
                        .assign(LL_l=lambda df: talib.MIN(df.low.values, road_period_l)) \
                        .assign(LC_l=lambda df: talib.MIN(df.close.values, road_period_l))\
                        .assign(max_HCL_l=lambda df: [max(row.HH_l - row.LC_l, row.HC_l - row.LL_l) for idx, row in df.iterrows()]) \
                        .assign(buyRange=lambda df: K1 * df.max_HCL_l.shift(1) + df.open)\
                        .assign(HH_s=lambda df: talib.MAX(df.high.values, road_period_s)) \
                        .assign(HC_s=lambda df: talib.MAX(df.close.values, road_period_s)) \
                        .assign(LL_s=lambda df: talib.MIN(df.low.values, road_period_s)) \
                        .assign(LC_s=lambda df: talib.MIN(df.close.values, road_period_s))\
                        .assign(max_HCL_s=lambda df: [max(row.HH_s - row.LC_s, row.HC_s - row.LL_s) for idx, row in df.iterrows()])\
                        .assign(sellRange=lambda df: df.open - K2 * df.max_HCL_s.shift(1)).reset_index()
                    buyRange = data.at[len(data) - 1, 'buyRange']  # 计算做多突破价格
                    sellRange = data.at[len(data) - 1, 'sellRange']  # 计算做空突破价格
                    c_ma = data.at[len(data) - 2, 'close'] - ma_1
                    ma_zd = ma_1 - ma_2

                    columns = ['stratid', 'exchange', 'symbol', 'period', 'optype', 'dir', 'starttime', 'endtime', 'price']

                    if (c_ma > 0) & (ma_zd > 0):
                        print('true')
                        buy_price = buyRange  # 触发价格，向上突破触发时买入
                        starttime = data.tickid.tolist()[-1]
                        endtime = data.tickid.tolist()[-1] + second_per_period

                        optype = 1
                        dir = 0

                        row = []
                        row.append(stratid)
                        row.append(exchange)
                        row.append(name)
                        row.append(period)
                        row.append(optype)
                        row.append(dir)
                        row.append(starttime)
                        row.append(endtime)
                        row.append(buy_price)

                        lst1.append(row)
                        try:
                            upsert_longshort_signal_future(stratid, exchange, name, period, optype, starttime, endtime, dir,
                                                           buy_price)
                        except Exception as e:
                            print(str(e))

                    if (c_ma < 0) & (ma_zd < 0):
                        print('true')
                        sell_price = sellRange  # 触发价格，向下突破触发时卖出
                        starttime = data.tickid.tolist()[-1]
                        endtime = data.tickid.tolist()[-1] + second_per_period

                        optype = 0
                        dir = 1

                        row = []
                        row.append(stratid)
                        row.append(exchange)
                        row.append(name)
                        row.append(period)
                        row.append(optype)
                        row.append(dir)
                        row.append(starttime)
                        row.append(endtime)
                        row.append(sell_price)

                        lst1.append(row)
                        try:
                            upsert_longshort_signal_future(stratid, exchange, name, period, optype, starttime, endtime, dir,
                                                           sell_price)
                        except Exception as e:
                            print(str(e))
            except Exception as e:
                sys.stderr.write(traceback.format_exc())

    print("=========================")
    df = pd.DataFrame(lst1, columns=columns)
    print(df)
    # df.to_csv('data/ocm_future.csv')
    print('true')


if __name__ == '__main__':
    # os.getcwd()
    # os.chdir(r'/Users/zhangfang/PycharmProjects/trading')
    # t0 = time.time()
    run_dt_future()

