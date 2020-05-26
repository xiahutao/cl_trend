# -*- coding: UTF-8 -*-
from __future__ import unicode_literals
import sys
sys.path.append('..')
import talib as tb
import json
import datetime,time
from backtest_func import *
import os
import matplotlib.pyplot as plt
import copy
import pandas as pd
from factors_gtja import *


def weight_Df(signal_df, pre_df, weight, loss_stop, win_stop, fee):
    signal_slect = copy.deepcopy(signal_df)
    pre_data = copy.deepcopy(pre_df)
    date_lst = copy.deepcopy(signal_slect.tickid.tolist())
    pre_data = pre_data.set_index(['tickid', 'symbol'])
    signal_slect_ = signal_slect.set_index(['tickid'])
    lst = []
    row_dict = {}
    row_dict['tickid'] = date_lst[0]
    row_dict['weight'] = 0
    row_dict['symbol'] = []
    row_dict['return'] = 0
    lst.append(row_dict)
    sig_state_lst = []
    for idx, row_ in signal_slect_.iterrows():
        today = idx
        code_lst = copy.deepcopy(row_.signal)
        date_left = [int(i) for i in date_lst if i > today]
        if (len(date_left) > 1) & (len(code_lst) > 0):
            for code in code_lst:

                if date_left[0] in pd.DataFrame(lst).set_index(['tickid']).index:
                    try:
                        if code == pd.DataFrame(lst).set_index(['tickid']).loc[date_left[0]]['symbol']:
                            # print('true1:', pd.DataFrame(lst).set_index(['tickid']).loc[date_left[0]]['symbol'])
                            continue
                    except Exception as e:
                        print(e)
                    try:
                        if code in pd.DataFrame(lst).set_index(['tickid']).loc[date_left[0]]['symbol'].tolist():
                            # print('true2:', pd.DataFrame(lst).set_index(['tickid']).loc[date_left[0]]['symbol'].tolist())
                            continue
                    except Exception as e:
                        print(e)
                code_hq_today = pre_data.loc[today, code]
                if code_hq_today['low1'] <= -loss_stop:
                    today_return = (1 - loss_stop) * (1 - fee) * (1 - fee) * weight
                    row_dict = {}
                    row_dict['tickid'] = date_left[0]
                    row_dict['weight'] = weight
                    row_dict['symbol'] = code
                    row_dict['return'] = today_return
                    lst.append(row_dict)
                    ret = (1 - loss_stop) * (1 - fee) * (1 - fee)-1
                    continue
                elif code_hq_today['high1'] >= win_stop:
                    today_return = (1 + win_stop) * (1 - fee) * (1 - fee) * weight
                    row_dict = {}
                    row_dict['tickid'] = date_left[0]
                    row_dict['weight'] = weight
                    row_dict['symbol'] = code
                    row_dict['return'] = today_return
                    lst.append(row_dict)
                    ret = (1 + win_stop) * (1 - fee) * (1 - fee)-1
                    continue
                elif code not in signal_slect_.loc[date_left[0]]['signal']:
                    today_return = (code_hq_today['pre1'] + 1) * (1 - fee) * (1 - fee) * weight
                    row_dict = {}
                    row_dict['tickid'] = date_left[0]
                    row_dict['weight'] = weight
                    row_dict['symbol'] = code
                    row_dict['return'] = today_return
                    lst.append(row_dict)
                    ret = (code_hq_today['pre1'] + 1) * (1 - fee) * (1 - fee)-1
                    continue
                today_return = (code_hq_today['pre1'] + 1) * weight * (1-fee)
                row_dict = {}
                row_dict['tickid'] = date_left[0]
                row_dict['weight'] = weight
                row_dict['symbol'] = code
                row_dict['return'] = today_return
                lst.append(row_dict)
                m = 1
                for tickid in date_left[1:]:

                    m = m + 1
                    m2 = m - 1
                    prenextname = 'pre' + str(m) + 'next'
                    prename = 'pre' + str(m2)
                    lowprename = 'low' + str(m)
                    highprename = 'high' + str(m)
                    prelst = []
                    prelst.append(0.0)
                    for n in range(1, m):
                        premaxname = 'pre' + str(n)
                        prelst.append(code_hq_today[premaxname])
                    if code_hq_today[highprename] >= win_stop:
                        today_return = (1 + win_stop) / (1+code_hq_today[prename]) * (1 - fee) * weight
                        row_dict = {}
                        row_dict['tickid'] = tickid
                        row_dict['weight'] = weight
                        row_dict['symbol'] = code
                        row_dict['return'] = today_return
                        lst.append(row_dict)
                        ret = (win_stop + 1) * (1 - fee) * (1 - fee)-1
                        break
                    elif (code_hq_today[lowprename] - max(prelst))/(1+max(prelst)) < -loss_stop:
                        today_return = (1+max(prelst)) * (1-loss_stop)/(1+code_hq_today[prename]) * weight * (1-fee)
                        row_dict = {}
                        row_dict['tickid'] = tickid
                        row_dict['weight'] = weight
                        row_dict['symbol'] = code
                        row_dict['return'] = today_return
                        lst.append(row_dict)
                        ret = ((max(prelst) + 1) * (1-loss_stop)) * (1 - fee) * (1 - fee)-1
                        break
                    elif code not in signal_slect_.loc[tickid]['signal']:
                        today_return = (code_hq_today[prenextname] + 1) * (1 - fee) * weight
                        row_dict = {}
                        row_dict['tickid'] = tickid
                        row_dict['weight'] = weight
                        row_dict['symbol'] = code
                        row_dict['return'] = today_return
                        lst.append(row_dict)
                        ret = (code_hq_today['pre' + str(m)] + 1) * (1 - fee) * (1 - fee)-1
                        break
                    else:
                        today_return = (code_hq_today[prenextname] + 1) * weight
                        row_dict = {}
                        row_dict['tickid'] = tickid
                        row_dict['weight'] = weight
                        row_dict['symbol'] = code
                        row_dict['return'] = today_return
                        lst.append(row_dict)
                        ret = (code_hq_today['pre' + str(m)] + 1) * (1 - fee) * (1 - fee)-1
                sig_state_row = []
                sig_state_row.append(today)
                sig_state_row.append(code)
                sig_state_row.append(ret)
                sig_state_row.append(m)
                sig_state_lst.append(sig_state_row)
    weight_df = pd.DataFrame(lst)\
        .assign(date_time=lambda df: df.tickid.apply(lambda x: datetime.datetime.fromtimestamp(x)))
    sig_state_df = pd.DataFrame(sig_state_lst, columns=['tickid', 'symbol', 'ret', 'holddays'])\
        .assign(date_time=lambda df: df.tickid.apply(lambda x: datetime.datetime.fromtimestamp(x)))
    return weight_df, sig_state_df


def net_Df(weight_df, signal_slect):
    weight_df = weight_df.assign(tickid=lambda df: df.tickid.apply(lambda x: int(x)))
    weight_df = weight_df.set_index(['tickid'])
    date_lst = [int(i) for i in copy.deepcopy(signal_slect.tickid.tolist())]
    net = 1
    lst = []
    for tickid in date_lst:
        if tickid in weight_df.index:
            change = weight_df.loc[tickid]['return'].sum() + (1-weight_df.loc[tickid]['weight'].sum())

            pos = len(weight_df.loc[tickid]['symbol'])
            if pos > 3:
                pos = 1
        else:
            change = 1.0
            pos = 0
        row_ = []
        row_.append(tickid)
        row_.append(change)
        net = net * change
        row_.append(net)
        row_.append(pos)
        lst.append(row_)
    net_df = pd.DataFrame(lst, columns=['tickid', 'change', 'net', 'pos'])\
        .assign(date_time=lambda df: df.tickid.apply(lambda x: datetime.datetime.fromtimestamp(x)))
    return net_df


if __name__ == '__main__':
    os.chdir(r'/Users/zhangfang/PycharmProjects')
    period = '1440m'
    n = 1440
    min_n_lst = [i for i in range(8, 9)]
    loss_stop_lst = [1]
    win_stop_lst = [100000]
    fee = 0.002
    date_lst = [('2017-10-01', '2019-04-01'), ('2017-10-01', '2018-01-01'), ('2018-01-01', '2018-07-01'),
                ('2018-07-01', '2019-12-01')]
    trade_lst = ['ethbtc', 'xrpbtc', 'ltcbtc', 'eosbtc', 'bchabcbtc', 'bchsvbtc', 'bnbbtc', 'trxbtc', 'adabtc',
                 'ontbtc', 'etcbtc']
    # trade_lst = ["ethbtc", "eosbtc", "xrpbtc", "trxbtc", "bchabcbtc", "bchsvbtc", "ontbtc", "ltcbtc", "adabtc",
    #              "bnbbtc"]
    name_lst = ['ethbtc', 'eosbtc', 'etcbtc', 'iotabtc', 'iostbtc', 'ltcbtc', 'neobtc', 'trxbtc', 'xrpbtc', 'xlmbtc',
                'adabtc', 'ontbtc', 'bnbbtc', 'bchabcbtc', 'bchsvbtc', "mdabtc", "stratbtc", "xmrbtc", "dashbtc",
                "xembtc", "zecbtc", "wavesbtc", "btgbtc", "vetbtc", "qtumbtc", "omgbtc", "zrxbtc", "gvtbtc"]

    # name_lst = ['ethbtc', 'eosbtc']
    # factor_lst = ["Alpha.alpha096", "Alpha.alpha112", "Alpha.alpha133", "Alpha.alpha052", "Alpha.alpha071",
    #               "Alpha.alpha128"]  # 正向因子
    # factor_lst = ["Alpha.alpha082", "Alpha.alpha049", "Alpha.alpha129", "Alpha.alpha093", "Alpha.alpha153",
    #               "Alpha.alpha046", "Alpha.alpha189"] # 反向因子
    factor_lst = ["Alpha.alpha003", "Alpha.alpha014", "Alpha.alpha050", "Alpha.alpha051", "Alpha.alpha069",
                  "Alpha.alpha128", "Alpha.alpha167", "Alpha.alpha175"]  # 正向因子001
    results = []
    df = pd.DataFrame(columns=['tickid', 'alpha'])
    c_ma_lst = []
    for symbol in name_lst:
        result_dict_temp = []
        data = pd.read_csv('data/' + symbol + '_' + period + '.csv').loc[
               :, ['tickid', 'close', 'high', 'low', 'open', 'volume']]
        c_ma = data[['tickid', 'close']]\
            .assign(c_ma=lambda df: df['close'] - tb.MA(df['close'].values, 25)).assign(symbol=symbol)[
            ['tickid', 'symbol', 'c_ma']]
        c_ma_lst.append(c_ma)
        for alpha in factor_lst:
            Alpha = Alphas(data)
            result_dict_temp.append(pd.DataFrame({'alpha': [alpha[6:]] * len(data), 'tickid': data.tickid.tolist(),
                                         symbol: eval(alpha)()}))

        df = df.merge(pd.concat(result_dict_temp), on=['tickid', 'alpha'], how='outer')
    df = df.sort_values(by=['alpha', 'tickid']).set_index(['alpha', 'tickid'])
    df = df.rank(axis=1, numeric_only=True, na_option="keep", ascending=True)
    # df.to_csv('cl_trend/data/signal.csv')
    c_ma_df = pd.concat(c_ma_lst)
    c_ma_df = c_ma_df[c_ma_df['c_ma'] > 0].set_index(['tickid', 'symbol'])
    index_lst = c_ma_df.index.tolist()
    pre_data_df = pd.read_hdf('data/coinbase_btc_28_predata' + period + '.h5', 'all')

    state_lst = []
    for min_n in min_n_lst:
        signal_lst = []
        for tickid, group in df.groupby(['tickid']):
            sers = group.sum()
            sers = sers[sers > 0]
            # max_symble = sers.idxmax()
            max_symble = list(sers.sort_values(ascending=False)[:min_n].index.values)
            # select_symble = [i for i in max_symble if i in trade_lst]
            select_symble = [i for i in max_symble if i in trade_lst and (tickid, i) in index_lst]
            signal_row = []
            signal_row.append(tickid)
            signal_row.append(select_symble)
            signal_lst.append(signal_row)
        signal_df = pd.DataFrame(signal_lst, columns=['tickid', 'signal'])
        # print(signal_df)
        signal_df = signal_df[signal_df['tickid'] >= 1507132800]\
            .assign(date_time=lambda df: df.tickid.apply(lambda x: datetime.datetime.fromtimestamp(x)))  # 2017-10-5
        signal_df.to_csv('cl_trend/data/signal_df001.csv')

        for loss_stop in loss_stop_lst:
            for win_stop in win_stop_lst:
                print(loss_stop, win_stop)
                weight_df, sig_state_df = weight_Df(signal_df, pre_data_df, 1/min_n, loss_stop, win_stop, fee)
                net_df = net_Df(weight_df, signal_df)
                # sig_state_df.to_csv('cl_trend/data/signal_state.csv')
                # weight_df.to_csv('cl_trend/data/weight_df.csv')
                # net_df.to_csv('cl_trend/data/net_df.csv')
                for (s_date, e_date) in date_lst:
                    net_df_ = net_df[(net_df['date_time'] >= s_date) & (net_df['date_time'] <= e_date)]\
                        .reset_index(drop=True)
                    net_df_[['date_time', 'net']].plot(
                        x='date_time', kind='line', grid=True,
                        title='001_' + period)
                    plt.show()
                    sig_state_df_ = sig_state_df[(sig_state_df['date_time'] >= s_date) & (sig_state_df['date_time'] <= e_date)] \
                        .reset_index(drop=True)

                    back_stime = net_df_.at[0, 'date_time']
                    back_etime = net_df_.at[len(net_df_) - 1, 'date_time']
                    net_lst = [i / net_df_.net.tolist()[0] for i in net_df_.net.tolist()]
                    ann_ROR = annROR(net_lst, n)
                    total_ret = net_lst[-1]-1
                    max_retrace = maxRetrace(net_lst, n)
                    sharp = yearsharpRatio(net_lst, n)
                    win_r, odds, ave_r, mid_r = get_winR_odds(sig_state_df_.ret.tolist())

                    state_row = []
                    state_row.append([i[-3:] for i in factor_lst])
                    state_row.append(n)
                    state_row.append(win_r)
                    state_row.append(odds)
                    state_row.append(total_ret)
                    state_row.append(ann_ROR)
                    state_row.append(sharp)
                    state_row.append(max_retrace)
                    state_row.append(len(sig_state_df_))
                    state_row.append(ave_r)
                    state_row.append(sig_state_df_.holddays.mean())

                    state_row.append(win_stop)
                    state_row.append(loss_stop)
                    state_row.append(min_n)

                    state_row.append(back_stime)
                    state_row.append(back_etime)
                    state_lst.append(state_row)
    res = pd.DataFrame(state_lst,
                       columns=['alpha', 'period', 'win_r', 'odds', 'total_ret', 'ann_ret', 'sharp', 'max_retrace',
                                'trade_times', 'ave_r', 'ave_hold_days', 'win_stop', 'loss_stop', 'max_hold_coins',
                                'back_stime', 'back_etime'])
    res.to_csv('cl_trend/data/alpha001_' + str(n) + '.csv')

