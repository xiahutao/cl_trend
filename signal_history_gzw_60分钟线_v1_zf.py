# coding=utf-8
'''
Created on 7.9, 2018

@author: fang.zhang
'''
from __future__ import division
from backtest_func import *
import matplotlib.pyplot as plt
from matplotlib import style
from dataapi import *
import pandas as pd
import talib

# style.use('ggplot')

if __name__ == '__main__':
    os.getcwd()
    print(os.path)
    os.chdir(r'/Users/zhangfang/PycharmProjects')
    t0 = time.time()
    # symble = 'yeeeth'
    x = 3
    symble_lst = 'btcusdt'
    n = 60  # 回测周期
    s_date = '2017-01-01'
    e_date = '2018-06-01'
    N1_lst = [4]  # 计数阈值
    N2_lst = [4]  # 统计周期
    N3_lst = [5]  # 计数上限
    N_ATR = 20
    ATR_n = 0.5

    fee = 0.001
    # s_date = '2016-01-01'
    # e_date = '2018-07-14'
    # group.loc[:, ['high', 'low', 'close', 'open']] = group.loc[:, ['high', 'low', 'close', 'open']]\
    #     .apply(lambda x: 1/x)
    # group = group.rename(columns={'high': 'low', 'low': 'high'})
    # print(group)
    df_lst = []
    lst = []
    state_lst = []
    group = pd.read_csv('data/btc_index_' + str(n) + 'm.csv').rename(columns={'date': 'date_time'}) \
        .assign(date_time=lambda df: df.date_time.apply(lambda x: str(x)))

    group = group[(group['date_time'] >= s_date) & (group['date_time'] <= e_date)] \
        .assign(day=lambda df: df.date_time.apply(lambda s: s[:10]))

    group_day = pd.read_csv('data/btc_index_' + '1440m' + '.csv').assign(
        date_time=lambda df: df.date_time + ' 00:00:00')
    group_day = group_day[(group_day['date_time'] >= s_date) & (group_day['date_time'] <= e_date)]
    group_day['atr'] = talib.ATR(group_day['high'].values, group_day['low'].values,
                                 group_day['close'].values, N_ATR)
    day_atr = group_day[['date_time', 'atr']] \
        .assign(atr=lambda df: df.atr.shift(1)) \
        .merge(group, on=['date_time'], how='right') \
        .sort_values(['date_time']).fillna(method='ffill').reset_index(drop=True)
    print(day_atr)

    for N1 in N1_lst:
        for N2 in N2_lst:
            for N3 in N3_lst:

                print(symble_lst)

                method = 'GZW' + '_' + str(N1) + '_' + str(N2) + '_' + str(N3)
                print(method)
                signal_lst = []
                trad_times = 0
                if len(group) > N2:
                    net = 1
                    net_lst = []
                    group_ = day_atr.assign(hhv=lambda df: talib.MAX(df.high.shift(1).values, N2)) \
                        .assign(llv=lambda df: talib.MIN(df.low.shift(1).values, N2)) \
                        .assign(hhv=lambda df: df.hhv.shift(1)) \
                        .assign(llv=lambda df: df.llv.shift(1)) \
                        .assign(close_1=lambda df: df.close.shift(1))
                    group_ = group_.dropna()

                    HHV_nums = 0
                    LLV_nums = 0

                    hhv_num_list = list()
                    llv_num_list = list()

                    for idx, irow in group_.iterrows():
                        if irow['close_1'] > irow['hhv']:
                            HHV_nums = HHV_nums + 1
                            LLV_nums = LLV_nums - 1
                        elif irow['close_1'] < irow['llv']:
                            HHV_nums = HHV_nums - 1
                            LLV_nums = LLV_nums + 1
                        hhv_num_list.append(HHV_nums)
                        llv_num_list.append(LLV_nums)
                    group_ = group_.assign(hhv_num=hhv_num_list) \
                        .assign(llv_num=llv_num_list) \
                        .assign(hhv_num_chg=lambda df: df.hhv_num - df.hhv_num.shift(N3)) \
                        .assign(llv_num_chg=lambda df: df.llv_num - df.llv_num.shift(N3))

                    position = 0
                    high_price_pre = 1000000000

                    for idx, _row in group_.iterrows():

                        if (position == 0) & (_row.hhv_num_chg > N1):
                            position = 1
                            s_time = _row.date_time
                            cost = max(_row.open, _row.low) * (1 + fee)
                            hold_price = []
                            high_price = []
                            hold_price.append(cost)
                            high_price.append(cost)
                            net = net * _row.close / cost
                        elif (position == 0) & (_row.high > high_price_pre):
                            position = 1
                            s_time = _row.date_time
                            cost = high_price_pre * (1 + fee)
                            if _row.open > high_price_pre:
                                cost = max(_row.open, _row.low) * (1 + fee)
                            hold_price = []
                            high_price = []
                            hold_price.append(cost)
                            high_price.append(cost)
                            net = net * _row.close / cost

                        elif position == 1:
                            if _row.low < max(hold_price) - _row.atr * ATR_n:
                                position = 0
                                trad_times += 1
                                high_price.append(_row.high)
                                e_time = _row.date_time
                                s_price = (max(hold_price) - _row.atr * ATR_n) * (1 - fee)
                                if min(_row.open, _row.high) < max(hold_price) - _row.atr * ATR_n:
                                    s_price = min(_row.open, _row.high) * (1 - fee)
                                high_price_pre = max(high_price)
                                ret = s_price / cost - 1
                                signal_row = []
                                signal_row.append(s_time)
                                signal_row.append(e_time)
                                signal_row.append(cost)
                                signal_row.append(s_price)
                                signal_row.append(ret)
                                signal_row.append(max(high_price) / cost - 1)
                                signal_row.append(len(hold_price))
                                net = net * s_price / _row.close_1 * (1 - fee)
                                signal_lst.append(signal_row)
                            else:
                                high_price.append(_row.high)
                                hold_price.append(_row.close)
                                net = net * _row.close / _row.close_1

                        net_lst.append(net)

                    ann_ROR = annROR(net_lst, n)
                    total_ret = net_lst[-1]
                    max_retrace = maxRetrace(net_lst)
                    sharp = yearsharpRatio(net_lst, n)

                    signal_state = pd.DataFrame(signal_lst,
                                                columns=['s_time', 'e_time', 'b_price', 's_price', 'ret',
                                                         'max_ret', 'hold_day']) \
                        .assign(method=method)
                    signal_state.to_csv('data/signal_gzw_' + str(n) + '_' + method + '.csv')
                    # df_lst.append(signal_state)
                    win_r, odds, ave_r, mid_r = get_winR_odds(signal_state.ret.tolist())
                    win_R_3, win_R_5, ave_max = get_winR_max(signal_state.max_ret.tolist())
                    state_row = []
                    state_row.append(symble_lst)
                    state_row.append(n)
                    state_row.append(win_r)
                    state_row.append(odds)
                    state_row.append(total_ret - 1)
                    state_row.append(ann_ROR)
                    state_row.append(sharp)
                    state_row.append(max_retrace)
                    state_row.append(len(signal_state))
                    state_row.append(ave_r)
                    state_row.append(signal_state.hold_day.mean())
                    state_row.append(mid_r)
                    state_row.append(win_R_3)
                    state_row.append(win_R_5)
                    state_row.append(ave_max)
                    state_row.append(N_ATR)
                    state_row.append(ATR_n)
                    state_row.append(N1)
                    state_row.append(N2)
                    state_row.append(N3)
                    state_lst.append(state_row)
                    # print('胜率=', win_r)
                    # print('盈亏比=', odds)
                    # print('总收益=', total_ret - 1)
                    # print('年化收益=', ann_ROR)
                    # print('夏普比率=', sharp)
                    # print('最大回撤=', max_retrace)
                    # print('交易次数=', len(signal_state))
                    # print('平均每次收益=', ave_r)
                    # print('平均持仓周期=', signal_state.hold_day.mean())
                    # print('中位数收益=', mid_r)
                    # print('超过3%胜率=', win_R_3)
                    # print('超过5%胜率=', win_R_5)
                    # print('平均最大收益=', ave_max)
                    # print('参数=', method)

    signal_state = pd.DataFrame(state_lst, columns=['symble', 'tm', 'win_r', 'odds', 'total_ret', 'ann_ret', 'sharp',
                                                    'max_retrace', 'trade_times', 'ave_r', 'ave_hold_days', 'mid_r',
                                                    'win_r_3', 'win_r_5', 'ave_max', 'art_N', 'art_n', 'n1',
                                                    'n2', 'n3'])
    print(signal_state)
    signal_state.to_csv('data/state_gzw' + '_' + str(n) + '_' + str(x) + '.csv')
#    signal_df = pd.concat(df_lst)
#    signal_df.to_csv('cl_trend/data/signal_tqa_' + str(n) + '_' + str(x) + '_.csv')
