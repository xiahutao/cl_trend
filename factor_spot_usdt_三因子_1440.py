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


if __name__ == '__main__':
    os.chdir(r'/Users/zhangfang/PycharmProjects')
    period = '1440m'
    n = 1440
    min_n_lst = [i for i in range(5, 6)]
    ma_period = 20
    loss_stop_lst = [1]
    win_stop_lst = [100000]
    fee = 0.002
    method = False
    date_lst = [('2018-01-01', '2019-04-01'), ('2018-01-01', '2018-07-01'), ('2018-07-01', '2019-12-01')]
    trade_lst = ['btcusdt', 'ethusdt', 'xrpusdt', 'ltcusdt', 'eosusdt', 'bchusdt', 'bchsvusdt', 'bnbusdt', 'trxusdt', 'adausdt',
                 'ontusdt', 'etcusdt']
    name_lst = ['btcusdt', 'ethusdt', 'eosusdt', 'etcusdt', 'iotausdt', 'iostusdt', 'ltcusdt', 'neousdt', 'trxusdt',
                'xrpusdt', 'xlmusdt', 'adausdt', 'ontusdt', 'bnbusdt', 'bchusdt', 'bchsvusdt', "xmrusdt", "dashusdt",
                "wavesusdt", "vetusdt", "qtumusdt", "zrxusdt"]

    # name_lst = ['ethbtc', 'eosbtc']
    factor_lst_all = ["Alpha.alpha112", "Alpha.alpha014", "Alpha.alpha071", "Alpha.alpha019", "Alpha.alpha031",
                      "Alpha.alpha020", "Alpha.alpha047", "Alpha.alpha072", "Alpha.alpha082", "Alpha.alpha153",
                      "Alpha.alpha046", "Alpha.alpha065"]  # 正向因子
    # factor_lst = ["Alpha.alpha082", "Alpha.alpha049", "Alpha.alpha129", "Alpha.alpha093", "Alpha.alpha046"]
    # factor_lst = ["Alpha.alpha093"]  # 正向因子001
    results = []

    c_ma_lst = []
    factor_lst_ = []
    for factor_1 in factor_lst_all:
        for factor_2 in factor_lst_all:
            for factor_3 in factor_lst_all:
                if (factor_1 < factor_2) & (factor_3 > factor_2):
                    factor_row = []
                    factor_row.append(factor_1)
                    factor_row.append(factor_2)
                    factor_row.append(factor_3)
                    factor_lst_.append(factor_row)
    # factor_lst_ = [["Alpha.alpha046", "Alpha.alpha049", "Alpha.alpha096", "Alpha.alpha129"],
    #                   ["Alpha.alpha049", "Alpha.alpha052", "Alpha.alpha071", "Alpha.alpha128"],
    #                   ["Alpha.alpha049", "Alpha.alpha052", "Alpha.alpha129", "Alpha.alpha133"],
    #                   ["Alpha.alpha049", "Alpha.alpha071", "Alpha.alpha129"],
    #                   ["Alpha.alpha049", "Alpha.alpha093", "Alpha.alpha096", "Alpha.alpha112"],
    #                   ["Alpha.alpha052", "Alpha.alpha071", "Alpha.alpha112"],
    #                   ["Alpha.alpha052", "Alpha.alpha093", "Alpha.alpha096", "Alpha.alpha133"],
    #                   ["Alpha.alpha052", "Alpha.alpha112", "Alpha.alpha128"],
    #                   ["Alpha.alpha052", "Alpha.alpha129", "Alpha.alpha133"],
    #                   ["Alpha.alpha071", "Alpha.alpha112", "Alpha.alpha128", "Alpha.alpha129"],
    #                   ["Alpha.alpha128", "Alpha.alpha133"],
    #                ["Alpha.alpha046"], ["Alpha.alpha049"], ["Alpha.alpha052"], ["Alpha.alpha071"],
    #                ["Alpha.alpha112"], ["Alpha.alpha128"]
    #                ]
    for symbol in name_lst:
        print(symbol)
        c_ma = pd.read_csv('data/' + symbol + '_1440m.csv').loc[:, ['tickid', 'close']] \
            .assign(ma=lambda df: tb.MA(df['close'].shift(1).values, ma_period - 1)).loc[:, ['tickid', 'ma']] \
            .merge(pd.read_csv('data/' + symbol + '_' + period + '.csv')[['tickid', 'close']], on=['tickid'], how='right') \
            .sort_values(['tickid']).fillna(method='ffill') \
            .assign(c_ma=lambda df: df.close - df.ma) \
            .assign(symbol=symbol)[['tickid', 'symbol', 'c_ma']]
        c_ma_lst.append(c_ma)
    c_ma_df = pd.concat(c_ma_lst)
    c_ma_df = c_ma_df[c_ma_df['c_ma'] > 0].set_index(['tickid', 'symbol'])
    index_lst = c_ma_df.index.tolist()
    # print(df)
    pre_data_df = pd.read_hdf('data/coinbase_usdt_28_predata' + period + '.h5', 'all')
    state_lst = []
    for factor_lst in factor_lst_:
        df = pd.DataFrame(columns=['tickid', 'alpha'])
        for symbol in name_lst:
            result_dict_temp = []
            data = pd.read_csv('data/' + symbol + '_' + period + '.csv').loc[
                   :, ['tickid', 'close', 'high', 'low', 'open', 'volume']]
            for alpha in factor_lst:
                Alpha = Alphas(data)
                result_dict_temp.append(pd.DataFrame({'alpha': [alpha[6:]] * len(data), 'tickid': data.tickid.tolist(),
                                             symbol: eval(alpha)()}))

            df = df.merge(pd.concat(result_dict_temp), on=['tickid', 'alpha'], how='outer')
        df = df.sort_values(by=['alpha', 'tickid']).set_index(['alpha', 'tickid'])
        df = df.rank(axis=1, numeric_only=True, na_option="keep", ascending=True)
        # df.to_csv('cl_trend/data/signal.csv')
        for min_n in min_n_lst:
            signal_lst = []
            for tickid, group in df.groupby(['tickid']):
                sers = group.sum()
                sers = sers[sers > 0]
                # max_symble = sers.idxmax()
                max_symble = list(sers.sort_values(ascending=method)[:min_n].index.values)
                # select_symble = [i for i in max_symble if i in trade_lst]
                select_symble = [i for i in max_symble if i in trade_lst and ((tickid, i) in index_lst)]
                signal_row = []
                signal_row.append(tickid)
                signal_row.append(select_symble)
                signal_lst.append(signal_row)
            signal_df = pd.DataFrame(signal_lst, columns=['tickid', 'signal'])
            # print(signal_df)
            signal_df = signal_df[signal_df['tickid'] >= 1507132800]\
                .assign(date_time=lambda df: df.tickid.apply(lambda x: datetime.datetime.fromtimestamp(x)))  # 2017-10-5
            # signal_df.to_csv('cl_trend/data/signal_df.csv')

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
                        # net_df_[['date_time', 'net']].plot(
                        #     x='date_time', kind='line', grid=True,
                        #     title='002_' + period)
                        # plt.show()
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
    res.to_csv('cl_trend/data/alpha_usdt_' + str(len(factor_lst_[0])) + '_' + str(n) + '.csv')

