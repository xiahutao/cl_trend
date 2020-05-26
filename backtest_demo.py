# coding:UTF-8
'''
Created on 9.6, 2017

@author: fang.zhang
'''
from __future__ import division
# from mssqlDBConnectorPeng import MysqlDBConnector, get_calen, getStockcode, getindexdaily
import pandas as pd
import urllib
import numpy as np
import math
import datetime
import time
import os
import requests
from threading import Thread
import path
import json
import logging
import matplotlib.pyplot as plt
import threading
from matplotlib import style

style.use('ggplot')


def maxRetrace(list):
    '''
    :param list:netlist
    :return: 最大历史回撤
    '''
    row = []
    for i in range(len(list)):
        row.append(1 - list[i] / max(list[:i + 1]))
    Max = max(row)
    return Max


def annROR(netlist):
    '''
    :param netlist:净值曲线
    :return: 年化收益
    '''
    return math.pow(netlist[-1] / netlist[0], 252 / len(netlist)) - 1


def daysharpRatio(netlist):
    row = []
    for i in range(1, len(netlist)):
        row.append(math.log(netlist[i] / netlist[i - 1]))
    return np.mean(row) / np.std(row)


def yearsharpRatio(netlist):
    row = []
    for i in range(1, len(netlist)):
        row.append(math.log(netlist[i] / netlist[i - 1]))
    return np.mean(row) / np.std(row) * math.pow(252, 0.5)


def netDf(weightdf, calen):
    weightdf['date_time'] = pd.to_datetime(weightdf['date_time'])
    net = 1
    list = []
    for tradedate in calen['date_time'].tolist():
        weightdfTD = weightdf[weightdf['date_time'] == tradedate]
        row = []
        if len(weightdfTD) != 0:
            change = np.float(sum(weightdfTD['holdingonedayreturn'])) + (1 - weightdfTD['weight'].sum()) * 1.00007033
        else:
            change = 1.0
        row.append(tradedate)
        row.append(change)
        net = net * change
        row.append(net)
        list.append(row)
    netdf = pd.DataFrame(list, columns=['date_time', 'change', 'net'])
    return netdf


def weightDf_atr(kslect, calen, weight, lossStop, winStop, maxholddays, fee):
    '''
    :param kslect:pre
    :return:df['Date Time', 'weight', 'STOCKCODE', 'holdingdayreturn']
    '''
    lst = []  # lst / dct
    weightdf = pd.DataFrame(lst, columns=['date_time', 'weight', 'stock_code', 'holdingdayreturn'])
    weightlist = []
    weightrow = []
    considerpositionlist = []
    weightrow.append(calen['date_time'].tolist()[0])
    weightrow.append(0)
    weightlist.append(weightrow)
    for tradedate in calen['date_time'].tolist():
        kslectTD = kslect[(kslect['date_time'] == tradedate)]
        # kslectTD=kslectTD.sort_values(by='winRatio',ascending=False)
        count = len(kslectTD)
        date = calen[calen['date_time'] > tradedate]
        if len(date) > 1:
            if count != 0:
                for code in kslectTD['stock_code'].tolist():
                    kslectTDcode = kslect[(kslect['date_time'] == tradedate) & (kslect['stock_code'] == code)]
                    weightbefore = weightdf[weightdf['date_time'] == date['date_time'].tolist()[0]]
                    code_lst = weightbefore.stock_code.tolist()
                    if code in code_lst:
                        continue
                    if weightbefore['weight'].sum() <= 1 - weight:
                        new_weight = weight
                    elif (weightbefore['weight'].sum() > 1 - weight) & (1 - weightbefore['weight'].sum() > weight / 1000):
                        new_weight = 1 - weightbefore['weight'].sum()
                    else:
                        continue
                    holdingdayreturn = (np.float(kslectTDcode['pre1']) + 1) * new_weight
                    row = []
                    row.append(date['date_time'].tolist()[0])
                    row.append(new_weight)
                    row.append(code)
                    row.append(holdingdayreturn)
                    lst.append(row)
                    if (np.float(kslectTDcode['low1']) > -lossStop) & (np.float(kslectTDcode['high1']) <= winStop):
                        m = 1

                        for Date in date['date_time'].tolist()[1:]:
                            row = []
                            m = m + 1
                            m2 = m - 1
                            prenextname = 'pre' + str(m) + 'next'
                            prename = 'pre' + str(m2)
                            lowprename = 'low' + str(m)
                            highprename = 'high' + str(m)
                            openprename = 'open' + str(m)
                            holdingdayreturn = (1 + np.float(kslectTDcode[prenextname])) * new_weight
                            prelist = []
                            prenextlist = []
                            prenextlist.append(kslectTDcode['pre1'])
                            for n in range(1, m):
                                premaxname = 'pre' + str(n)
                                prenextmaxname = 'pre' + str(n) + 'next'
                                prelist.append(np.float(kslectTDcode[premaxname]))
                                if n > 1:
                                    prenextlist.append(np.float(kslectTDcode[prenextmaxname]))
                            prelist.append(0.0)
                            if ((np.float(kslectTDcode[lowprename]) - max(prelist)) / (
                                        1 + max(prelist)) > -lossStop) & (
                                np.float(kslectTDcode[highprename]) < winStop):
                                row.append(Date)
                                row.append(new_weight)
                                row.append(code)
                                if m < maxholddays:
                                    row.append(holdingdayreturn)
                                # allreturn = np.float(kslectTDcode['pre' + str(m)])
                                else:
                                    row.append(holdingdayreturn * (1 - fee))
                                lst.append(row)
                                allreturn = np.float(kslectTDcode['pre' + str(m)])
                                if m > maxholddays:
                                    break
                            elif np.float(kslectTDcode[highprename]) >= winStop:
                                ret = win_stop
                                if np.float(kslectTDcode[openprename]) >= winStop:
                                    ret = np.float(kslectTDcode[openprename])
                                row.append(Date)
                                row.append(new_weight)
                                row.append(code)
                                row.append(
                                    (1 + ret) / (1 + np.float(kslectTDcode[prename])) * new_weight * (1 - fee))
                                lst.append(row)
                                allreturn = ret
                                break

                            elif (np.float(kslectTDcode[lowprename]) - max(prelist)) / (
                                        1 + max(prelist)) <= -lossStop:
                                ret = (np.float(kslectTDcode[lowprename]) - max(prelist)) / (
                                        1 + max(prelist))
                                if (np.float(kslectTDcode[openprename]) - max(prelist)) / (
                                        1 + max(prelist)) <= -lossStop:
                                    ret = (np.float(kslectTDcode[openprename]) - max(prelist)) / (
                                        1 + max(prelist))
                                row.append(Date)
                                row.append(new_weight)
                                row.append(code)
                                row.append((1 + max(prelist)) / (1 + np.float(kslectTDcode[prename])) * (
                                    1 + ret) * new_weight * (1 - fee))
                                lst.append(row)
                                allreturn = (1 + max(prelist)) * (1 + ret) - 1
                                break
                            else:
                                break
                        considerpositionrow = []
                        considerpositionrow.append(tradedate)
                        considerpositionrow.append(code)
                        considerpositionrow.append(allreturn)
                        considerpositionrow.append(m)
                        considerpositionlist.append(considerpositionrow)

                    elif (np.float(kslectTDcode['low1']) <= -lossStop) | (np.float(kslectTDcode['high1']) > winStop):
                        row = []
                        row.append(date['date_time'].tolist()[1])
                        row.append(new_weight)
                        row.append(code)
                        row.append((1 + kslectTDcode['open2'].tolist()[0])/(1 + kslectTDcode['pre1'].tolist()[0]) * new_weight * (1 - fee))
                        allreturn = np.float(kslectTDcode['open2'])
                        lst.append(row)
                        considerpositionrow = []
                        considerpositionrow.append(tradedate)
                        considerpositionrow.append(code)
                        considerpositionrow.append(allreturn)
                        considerpositionrow.append(2)
                        considerpositionlist.append(considerpositionrow)

                    weightdf = pd.DataFrame(lst, columns=['date_time', 'weight', 'stock_code', 'holdingonedayreturn'])
                    # weightdf['date_time'] = pd.to_datetime(weightdf['date_time'])
            weighttoday = weightdf[weightdf['date_time'] == date['date_time'].tolist()[0]]
            weightrow = []
            weightrow.append(date['date_time'].tolist()[0])
            weightrow.append(weighttoday['weight'].sum())
            weightlist.append(weightrow)
    postdf = pd.DataFrame(weightlist, columns=['date_time', 'position'])
    considerpositiondf = pd.DataFrame(considerpositionlist,
                                      columns=['date_time', 'stock_code', 'return', 'hold_days'])
    return weightdf, postdf, considerpositiondf


def weightDf(kslect, calen, weight, lossStop, winStop, maxholddays, fee):
    '''
    :param kslect:pre
    :return:df['Date Time', 'weight', 'STOCKCODE', 'holdingdayreturn']
    '''
    lst = []  # lst / dct
    weightdf = pd.DataFrame(lst, columns=['date_time', 'weight', 'stock_code', 'holdingdayreturn'])
    weightlist = []
    weightrow = []
    considerpositionlist = []
    weightrow.append(calen['date_time'].tolist()[0])
    weightrow.append(0)
    weightlist.append(weightrow)
    count_lst = []
    for tradedate in calen['date_time'].tolist():
        kslectTD = kslect[(kslect['date_time'] == tradedate)]
        # kslectTD=kslectTD.sort_values(by='winRatio',ascending=False)
        count = len(kslectTD)
        count_row = []
        count_row.append(tradedate)
        count_row.append(count)
        count_lst.append(count_row)
        date = calen[calen['date_time'] > tradedate]
        if len(date) > 1:
            if count != 0:
                for code in kslectTD['stock_code'].tolist():
                    kslectTDcode = kslect[(kslect['date_time'] == tradedate) & (kslect['stock_code'] == code)]
                    weightbefore = weightdf[weightdf['date_time'] == date['date_time'].tolist()[0]]
                    code_lst = weightbefore.stock_code.tolist()
                    if code in code_lst:
                        continue
                    if weightbefore['weight'].sum() <= 1 - weight:
                        new_weight = weight
                    elif (weightbefore['weight'].sum() > 1 - weight) & (1 - weightbefore['weight'].sum() > weight / 1000):
                        new_weight = 1 - weightbefore['weight'].sum()
                    else:
                        continue
                    if np.float(kslectTDcode['pre1']) > -10:
                        holdingdayreturn = (np.float(kslectTDcode['pre1']) + 1) * new_weight
                        row = []
                        row.append(date['date_time'].tolist()[0])
                        row.append(new_weight)
                        row.append(code)
                        row.append(holdingdayreturn)
                        lst.append(row)
                    else:
                        print('tingpai')
                        continue
                    if (np.float(kslectTDcode['low1']) > -lossStop) & (np.float(kslectTDcode['high1']) <= winStop):
                        m = 1

                        for Date in date['date_time'].tolist()[1:]:
                            row = []
                            m = m + 1
                            m2 = m - 1
                            prenextname = 'pre' + str(m) + 'next'
                            prename = 'pre' + str(m2)
                            lowprename = 'low' + str(m)
                            highprename = 'high' + str(m)
                            openprename = 'open' + str(m)
                            holdingdayreturn = (1 + np.float(kslectTDcode[prenextname])) * new_weight
                            prelist = []
                            prenextlist = []
                            prenextlist.append(kslectTDcode['pre1'])
                            for n in range(1, m):
                                premaxname = 'pre' + str(n)
                                prenextmaxname = 'pre' + str(n) + 'next'
                                prelist.append(np.float(kslectTDcode[premaxname]))
                                if n > 1:
                                    prenextlist.append(np.float(kslectTDcode[prenextmaxname]))
                            prelist.append(0.0)
                            if ((np.float(kslectTDcode[lowprename]) - max(prelist)) / (
                                        1 + max(prelist)) > -lossStop) & (
                                np.float(kslectTDcode[highprename]) < winStop):
                                row.append(Date)
                                row.append(new_weight)
                                row.append(code)
                                if m < maxholddays:
                                    row.append(holdingdayreturn)
                                # allreturn = np.float(kslectTDcode['pre' + str(m)])
                                else:
                                    row.append(holdingdayreturn * (1 - fee))
                                lst.append(row)
                                if m >= maxholddays:
                                    allreturn = np.float(kslectTDcode['pre' + str(m)])
                                    break
                            elif np.float(kslectTDcode[highprename]) >= winStop:
                                ret = winStop
                                if np.float(kslectTDcode[openprename]) >= winStop:
                                    ret = np.float(kslectTDcode[openprename])
                                row.append(Date)
                                row.append(new_weight)
                                row.append(code)
                                row.append(
                                    (1 + ret) / (1 + np.float(kslectTDcode[prename])) * new_weight * (1 - fee))
                                lst.append(row)
                                allreturn = ret
                                break

                            elif (np.float(kslectTDcode[lowprename]) - max(prelist)) / (
                                        1 + max(prelist)) <= -lossStop:
                                ret = (np.float(kslectTDcode[lowprename]) - max(prelist)) / (
                                        1 + max(prelist))
                                if (np.float(kslectTDcode[openprename]) - max(prelist)) / (
                                        1 + max(prelist)) <= -lossStop:
                                    ret = (np.float(kslectTDcode[openprename]) - max(prelist)) / (
                                        1 + max(prelist))
                                row.append(Date)
                                row.append(new_weight)
                                row.append(code)
                                row.append((1 + max(prelist)) / (1 + np.float(kslectTDcode[prename])) * (
                                    1 + ret) * new_weight * (1 - fee))
                                lst.append(row)
                                allreturn = (1 + max(prelist)) * (1 + ret) - 1
                                break
                            else:
                                break
                        considerpositionrow = []
                        considerpositionrow.append(tradedate)
                        considerpositionrow.append(code)
                        considerpositionrow.append(allreturn)
                        considerpositionrow.append(m)
                        considerpositionlist.append(considerpositionrow)

                    elif (np.float(kslectTDcode['low1']) <= -lossStop) | (np.float(kslectTDcode['high1']) > winStop):
                        row = []
                        row.append(date['date_time'].tolist()[1])
                        row.append(new_weight)
                        row.append(code)
                        row.append((1 + kslectTDcode['open2'].tolist()[0])/(1 + kslectTDcode['pre1'].tolist()[0]) * new_weight * (1 - fee))
                        allreturn = np.float(kslectTDcode['open2'])
                        lst.append(row)
                        considerpositionrow = []
                        considerpositionrow.append(tradedate)
                        considerpositionrow.append(code)
                        considerpositionrow.append(allreturn)
                        considerpositionrow.append(2)
                        considerpositionlist.append(considerpositionrow)

                    weightdf = pd.DataFrame(lst, columns=['date_time', 'weight', 'stock_code', 'holdingonedayreturn'])
                    # weightdf['date_time'] = pd.to_datetime(weightdf['date_time'])
            weighttoday = weightdf[weightdf['date_time'] == date['date_time'].tolist()[0]]
            weightrow = []
            weightrow.append(date['date_time'].tolist()[0])
            weightrow.append(weighttoday['weight'].sum())
            weightlist.append(weightrow)
    postdf = pd.DataFrame(weightlist, columns=['date_time', 'position'])
    considerpositiondf = pd.DataFrame(considerpositionlist,
                                      columns=['date_time', 'stock_code', 'return', 'hold_days'])
    signal_count = pd.DataFrame(count_lst, columns=['date_time', 'signal_count'])
    return weightdf, postdf, considerpositiondf, signal_count


def get_predata(stockhqdata, maxholdday):
    df_lst = []
    for name, group in stockhqdata.groupby('stock_code'):
        t0 = time.time()
        if group.shape[0] < 20:
            print("Stock {stk} has only {cnt} data and skip it.".format(stk=name, cnt=group.shape[0]))
            continue
        group = group.set_index('stock_code')
        _df_lst = [group, ]
        df_shift_1 = group.loc[:, ['open', ]].shift(-1)
        for p in range(1, maxholdday + 1, 1):
            df_shift = group.loc[:, ['close', 'low', 'high', 'open']].shift(-p).rename(
                columns={"close": "c_p", 'low': "l_p", "high": "h_p", "open": 'o_p'})
            df_shift_ = group.loc[:, ['close', ]].shift(-p + 1).rename(columns={"close": "c_p_"})
            _df = pd.concat([df_shift_1, df_shift, df_shift_], axis=1) \
                .assign(pre=lambda df: (df.c_p / df.open) - 1) \
                .assign(pre_next=lambda df: (df.c_p / df.c_p_ - 1)) \
                .assign(low=lambda df: (df.l_p / df.open - 1)) \
                .assign(open_n=lambda df: (df.o_p / df.open - 1)) \
                .assign(high=lambda df: (df.h_p / df.open - 1)) \
                .rename(
                columns={'pre': 'pre{p}'.format(p=p), 'pre_next': 'pre{p}next'.format(p=p), 'low': 'low{p}'.format(p=p),
                         'high': 'high{p}'.format(p=p), 'open_n': 'open{p}'.format(p=p)}) \
                .drop(['c_p', 'l_p', 'h_p', 'open', 'c_p_', 'o_p'], axis=1)
            _df_lst.append(_df)
        df = pd.concat(_df_lst, axis=1)
        df_lst.append(df)
        print(time.time() - t0)
    dealdataall = pd.concat(df_lst)
    dealdataall = dealdataall.reset_index(drop=False)
    return dealdataall


def weightDf_with_sell(kslect, sell_slect, calen, weight, lossStop, winStop, maxholddays, fee):
    '''
    :param kslect:pre
    :return:df['Date Time', 'weight', 'STOCKCODE', 'holdingdayreturn']
    '''
    lst = []  # lst / dct
    weightdf = pd.DataFrame(lst, columns=['date_time', 'weight', 'stock_code', 'holdingdayreturn'])
    weightlist = []
    weightrow = []
    considerpositionlist = []
    weightrow.append(calen['date_time'].tolist()[0])
    weightrow.append(0)
    weightlist.append(weightrow)
    count_lst = []
    sell_slect = sell_slect.set_index(['stock_code', 'date_time_1'])
    for tradedate in calen['date_time'].tolist():
        kslectTD = kslect[(kslect['date_time'] == tradedate)]
        # kslectTD=kslectTD.sort_values(by='winRatio',ascending=False)
        count = len(kslectTD)
        count_row = []
        count_row.append(tradedate)
        count_row.append(count)
        count_lst.append(count_row)
        date = calen[calen['date_time'] > tradedate]
        if len(date) > 1:
            if count != 0:
                for code in kslectTD['stock_code'].tolist():
                    kslectTDcode = kslect[(kslect['date_time'] == tradedate) & (kslect['stock_code'] == code)]
                    weightbefore = weightdf[weightdf['date_time'] == date['date_time'].tolist()[0]]
                    code_lst = weightbefore.stock_code.tolist()
                    if code in code_lst:
                        continue
                    if weightbefore['weight'].sum() <= 1 - weight:
                        new_weight = weight
                    elif (weightbefore['weight'].sum() > 1 - weight) & (1 - weightbefore['weight'].sum() > weight / 1000):
                        new_weight = 1 - weightbefore['weight'].sum()
                    else:
                        continue
                    if np.float(kslectTDcode['pre1']) > -10:
                        holdingdayreturn = (np.float(kslectTDcode['pre1']) + 1) * new_weight
                        row = []
                        row.append(date['date_time'].tolist()[0])
                        row.append(new_weight)
                        row.append(code)
                        row.append(holdingdayreturn)
                        lst.append(row)
                    else:
                        print('tingpai')
                        continue
                    if (np.float(kslectTDcode['low1']) > -lossStop) & (np.float(kslectTDcode['high1']) <= winStop):
                        m = 1
                        for Date in date['date_time'].tolist()[1:]:
                            m = m + 1
                            m2 = m - 1
                            prenextname = 'pre' + str(m) + 'next'
                            prename = 'pre' + str(m2)
                            lowprename = 'low' + str(m)
                            highprename = 'high' + str(m)
                            openprename = 'open' + str(m)
                            holdingdayreturn = (1 + np.float(kslectTDcode[prenextname])) * new_weight
                            prelist = []
                            prenextlist = []
                            prenextlist.append(kslectTDcode['pre1'])
                            for n in range(1, m):
                                premaxname = 'pre' + str(n)
                                prenextmaxname = 'pre' + str(n) + 'next'
                                prelist.append(np.float(kslectTDcode[premaxname]))
                                if n > 1:
                                    prenextlist.append(np.float(kslectTDcode[prenextmaxname]))
                            prelist.append(0.0)
                            if (code, Date) in sell_slect.index.values.tolist():

                                row = []
                                row.append(Date)
                                row.append(new_weight)
                                row.append(code)
                                row.append((np.float(kslectTDcode[openprename]) + 1) / (
                                np.float(kslectTDcode[prename]) + 1) * new_weight)
                                lst.append(row)
                                allreturn = np.float(kslectTDcode[openprename])
                                m = m - 1
                                break
                            elif ((np.float(kslectTDcode[lowprename]) - max(prelist)) / (
                                        1 + max(prelist)) > -lossStop) & (
                                np.float(kslectTDcode[highprename]) < winStop):
                                row = []
                                row.append(Date)
                                row.append(new_weight)
                                row.append(code)
                                if m < maxholddays:
                                    row.append(holdingdayreturn)
                                # allreturn = np.float(kslectTDcode['pre' + str(m)])
                                else:
                                    row.append(holdingdayreturn * (1 - fee))
                                lst.append(row)
                                if m >= maxholddays:
                                    allreturn = np.float(kslectTDcode['pre' + str(m)])
                                    break
                            elif np.float(kslectTDcode[highprename]) >= winStop:
                                ret = winStop
                                if np.float(kslectTDcode[openprename]) >= winStop:
                                    ret = np.float(kslectTDcode[openprename])
                                row = []
                                row.append(Date)
                                row.append(new_weight)
                                row.append(code)
                                row.append(
                                    (1 + ret) / (1 + np.float(kslectTDcode[prename])) * new_weight * (1 - fee))
                                lst.append(row)
                                allreturn = ret
                                break

                            elif (np.float(kslectTDcode[lowprename]) - max(prelist)) / (
                                        1 + max(prelist)) <= -lossStop:
                                ret = (np.float(kslectTDcode[lowprename]) - max(prelist)) / (
                                        1 + max(prelist))
                                if (np.float(kslectTDcode[openprename]) - max(prelist)) / (
                                        1 + max(prelist)) <= -lossStop:
                                    ret = (np.float(kslectTDcode[openprename]) - max(prelist)) / (
                                        1 + max(prelist))
                                row = []
                                row.append(Date)
                                row.append(new_weight)
                                row.append(code)
                                row.append((1 + max(prelist)) / (1 + np.float(kslectTDcode[prename])) * (
                                    1 + ret) * new_weight * (1 - fee))
                                lst.append(row)
                                allreturn = (1 + max(prelist)) * (1 + ret) - 1
                                break
                            else:
                                break
                        considerpositionrow = []
                        considerpositionrow.append(tradedate)
                        considerpositionrow.append(code)
                        considerpositionrow.append(allreturn)
                        considerpositionrow.append(m)
                        considerpositionlist.append(considerpositionrow)

                    elif (np.float(kslectTDcode['low1']) <= -lossStop) | (np.float(kslectTDcode['high1']) > winStop):
                        row = []
                        row.append(date['date_time'].tolist()[1])
                        row.append(new_weight)
                        row.append(code)
                        row.append((1 + kslectTDcode['open2'].tolist()[0])/(1 + kslectTDcode['pre1'].tolist()[0]) * new_weight * (1 - fee))
                        allreturn = np.float(kslectTDcode['open2'])
                        lst.append(row)
                        considerpositionrow = []
                        considerpositionrow.append(tradedate)
                        considerpositionrow.append(code)
                        considerpositionrow.append(allreturn)
                        considerpositionrow.append(2)
                        considerpositionlist.append(considerpositionrow)

                    weightdf = pd.DataFrame(lst, columns=['date_time', 'weight', 'stock_code', 'holdingonedayreturn'])
                    # weightdf['date_time'] = pd.to_datetime(weightdf['date_time'])
            weighttoday = weightdf[weightdf['date_time'] == date['date_time'].tolist()[0]]
            weightrow = []
            weightrow.append(date['date_time'].tolist()[0])
            weightrow.append(weighttoday['weight'].sum())
            weightlist.append(weightrow)
    postdf = pd.DataFrame(weightlist, columns=['date_time', 'position'])
    considerpositiondf = pd.DataFrame(considerpositionlist,
                                      columns=['date_time', 'stock_code', 'return', 'hold_days'])
    signal_count = pd.DataFrame(count_lst, columns=['date_time', 'signal_count'])
    return weightdf, postdf, considerpositiondf, signal_count


if __name__ == '__main__':
    os.getcwd()
    os.chdir(r'F:\\PycharmProjects\\cl_zjm')
    allstarttime = '20160101'
    s_date = '20160115'
    e_date = '20170701'
    list_enddate = '20150701'
    x = 0
    weight_lst = [0.02]
    win_lst = [100, 0.2, 0.15, 0.1]
    loss_lst = [0.2, 0.15, 0.1, 0.05, 1]
    max_day_lst = [20]
    fee = 0.002
    index_code = ['000001']
    csd_znxg = MysqlDBConnector('db_znxg')
    csd_hq = MysqlDBConnector('db_hq')
    today1 = datetime.date.today()
    print(today1)
    transferstr = lambda x: datetime.datetime.strftime(x, '%Y%m%d')
    today = datetime.datetime.strftime(today1, '%Y%m%d')
    print(today)
    calen = get_calen(s_date, e_date).rename(columns={'day': 'date_time'})
    print(calen)
    signal_all = pd.read_hdf('data/ddtj_signal_price_chng_' + str(x) + '_s.h5', 'all')[
        ['date_time', 'stock_code', 'method', 'type']]
    print(signal_all)
    signal_all = signal_all[(signal_all['date_time'] < '20170701') & (signal_all['type'] == 1)]\
        .drop('type', axis=1)
    # signal_all = signal_all[signal_all['date_time'] < '20170701']
    # date_lst = set(list(signal_all.date_time))
    # non_signal_rate = len(date_lst) / len(calen)
    # print(non_signal_rate)
    signal_all = signal_all.merge(pd.read_hdf('data/pre_data_160101_180101.h5', 'all'), on=['date_time', 'stock_code'])

    index_hq = getindexdaily(index_code, s_date, e_date)[['date_time', 'close']].reset_index()\
        .assign(zs_net=lambda df: df.close/df.at[0, 'close'])[['date_time', 'zs_net']]

    print(index_hq)
    state_lst = []
    for method_, group in signal_all.groupby(['method']):
        if len(group) < 100:
            continue
        for weight in weight_lst:
            for win_stop in win_lst:
                for loss_stop in loss_lst:
                    for max_days in max_day_lst:
                        method = method_ + '_' + str(weight) + '_' + str(loss_stop) + '_' + str(win_stop) + '_' + str(max_days)
                        weightdf, postdf, considerpositiondf, countdf = weightDf(group, calen, weight, loss_stop, win_stop,
                                                                        max_days, fee)
                        # considerpositiondf.to_csv('data/considerpositiondf.csv')

                        win = considerpositiondf[considerpositiondf['return'] > 0]
                        loss = considerpositiondf[considerpositiondf['return'] < 0]
                        win_r = len(win) / (len(win) + len(loss))
                        odds = -np.mean(win['return'])/np.mean(loss['return'])
                        ave_holddays = np.mean(considerpositiondf.hold_days)
                        net = netDf(weightdf, calen)\
                            # .merge(index_hq, on=['date_time'])\
                            # .assign(zs_net=lambda df: df.zs_net.apply(lambda x: np.float(x)))
                        # net[['date_time', 'net', 'zs_net']].plot(x='date_time', kind='line', grid=True)
                        # plt.show()
                        max_retrace = maxRetrace(net.net.tolist())
                        print(max_retrace)
                        yet = annROR(net.net.tolist())
                        print(yet)
                        ave_position = np.mean(postdf.position)
                        non_position = postdf[postdf['position'] == 0]
                        sharp = yearsharpRatio(net.net.tolist())
                        state_row = []
                        state_row.append(method)
                        state_row.append(len(group))
                        state_row.append(net.net.tolist()[-1]-1)
                        state_row.append(yet)
                        state_row.append(max_retrace)
                        state_row.append(ave_position)
                        state_row.append(sharp)
                        state_row.append(len(non_position)/len(postdf))
                        state_row.append(ave_holddays)
                        state_row.append(win_r)
                        state_row.append(odds)
                        state_row.append(max(considerpositiondf['return']))
                        state_row.append(min(considerpositiondf['return']))
                        state_row.append(np.mean(considerpositiondf['return']))
                        state_lst.append(state_row)
    stat_df = pd.DataFrame(state_lst, columns=['method', 'count', 'return', 'yeild', 'max_retrace', 'ave_position', 'sharp',
                                               'non_positon_rate', 'ave_holddays', 'win_R', 'odds', 'max_ret',
                                               'min_ret', 'ave_ret'])
    print(stat_df)
    stat_df.to_csv('data/position_stat_' + str(x) + '.csv')
