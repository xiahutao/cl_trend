# coding=utf-8
'''
Created on 7.9, 2018

@author: fang.zhang
'''
from __future__ import division
from backtest_func import *
import matplotlib.pyplot as plt
from matplotlib import style
import talib

style.use('ggplot')


if __name__ == '__main__':
    os.getcwd()
    print(os.path)
    os.chdir(r'/Users/zhangfang/PycharmProjects')
    t0 = time.time()
    s_time = '2018-01-01 00:00:00'  # 回测开始日期
    e_time = '2018-08-20 00:00:00'  # 回测结束日期
    csd = MysqlDBConnector('production')
    exchange = 'HUOBIPRO'  # 交易所名称,取火币数据
    period_lst = ['15min']  # ['15min', '30min', '60min', '4hour']  # 回测周期
    coin_couple_lst = [('btc', 'usdt')]  # 币对

    fee = 0.004  # 交易费用

    #  定义参数
    N1_lst = [20]  # 均线周期
    P_ATR_LST = [20]  # ATR周期
    ATR_n_lst = [0.5]  # ATR倍数
    road_period_lst = [5]  # 唐奇安通道周期
    param_lst = [N1_lst, P_ATR_LST, ATR_n_lst, road_period_lst]  # 参数列表

    df_lst = []
    lst = []
    state_lst = []
    for coin_couple in coin_couple_lst:
        data_day = get_market_kline(get_symble_id(exchange, coin_couple[0], coin_couple[1]), '1day', s_time, e_time) \
                       .loc[:, ['tickid', 'date_time', 'high', 'low', 'close', 'open', 'volume']]  # 从数据库取日级别数据
        print(data_day)
        for N1 in N1_lst:
            for N_ATR in P_ATR_LST:
                data_day['atr'] = talib.ATR(data_day['high'].values, data_day['low'].values,
                                            data_day['close'].values, N_ATR)
                data_day['ma'] = talib.MA(data_day['close'].values, N1)
                for period in period_lst:
                    n = get_min_n_from_period(period)
                    data = get_market_kline(get_symble_id(exchange, coin_couple[0], coin_couple[1]), period, s_time, e_time) \
                               .loc[:, ['tickid', 'date_time', 'high', 'low', 'close', 'open', 'volume']]  # 从数据库取数据
                    print(data)

                    day_atr = data_day[['date_time', 'atr', 'ma']] \
                        .assign(ma_zd=lambda df: df.ma.shift(1) - df.ma.shift(2)) \
                        .assign(atr=lambda df: df.atr.shift(1)) \
                        .merge(data, on=['date_time'], how='right') \
                        .sort_values(['date_time']).fillna(method='ffill').drop(['index'], axis=1)

                    for ATR_n in ATR_n_lst:
                        for road_period in road_period_lst:
                            method = str(N1) + '_' + str(N_ATR) + '_' + str(ATR_n) + '_' + str(road_period)
                            signal_lst = []
                            trad_times = 0

                            if len(day_atr) > 20:
                                net = 1
                                net_lst = []
                                day_atr['h10'] = talib.MAX(day_atr['high'].values, road_period)
                                day_atr['l10'] = talib.MIN(day_atr['low'].values, road_period)
                                day_atr = day_atr.assign(close_1=lambda df: df.close.shift(1)
                                                         ) \
                                    .assign(h10=lambda df: df.h10.shift(1)) \
                                    .assign(l10=lambda df: df.l10.shift(1)) \
                                    .assign(c_ma=lambda df: df.close_1 - df.ma.shift(1))
                                day_atr.to_csv('cl_trend/day_atr.csv')
                                position = 0
                                high_price_pre = 1000000
                                # signal_row = []
                                # stock_row = []
                                for idx, _row in day_atr.iterrows():

                                    if (position == 0) & (_row.high > _row.h10) & (_row.c_ma > 0) & (_row.ma_zd > 0):
                                        position = 1
                                        s_time = _row.date_time
                                        cost = _row.h10
                                        hold_price = []
                                        high_price = []
                                        hold_price.append(cost)
                                        high_price.append(cost)
                                        net = net * _row.close / cost
                                    elif (position == 0) & (_row.high > high_price_pre):
                                        position = 1
                                        s_time = _row.date_time
                                        cost = high_price_pre
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
                                            s_price = max(hold_price) - _row.atr * ATR_n
                                            high_price_pre = max(high_price)
                                            ret = s_price / cost - 1
                                            signal_row = []
                                            signal_row.append(s_time)
                                            signal_row.append(e_time)
                                            signal_row.append(cost)
                                            signal_row.append(s_price)
                                            signal_row.append(ret - fee)
                                            signal_row.append(max(high_price) / cost - 1)
                                            signal_row.append(len(hold_price))
                                            net = net * s_price/_row.close_1 * (1 - fee)
                                            signal_lst.append(signal_row)
                                        else:
                                            high_price.append(_row.high)
                                            hold_price.append(_row.close)
                                            net = net * _row.close / _row.close_1
                                    net_row = []
                                    net_row.append(_row.date_time)
                                    net_row.append(net)
                                    net_lst.append(net_row)
                                net_df = pd.DataFrame(net_lst, columns=['date_time', 'net']) \
                                    .merge(data.loc[:, ['date_time', 'close']]).assign(
                                    close=lambda df: df.close / df.close.tolist()[0])
                                net_df.to_csv('cl_trend/data/net_' + coin_couple[0] + coin_couple[1] + '.csv')
                                net_df[['date_time', 'net', 'close']].plot(
                                    x='date_time', kind='line', grid=True,
                                                                           title=coin_couple[0] + coin_couple[1] + '_' + period)
                                plt.xlabel('date_time')
                                plt.savefig('cl_trend/data/' + coin_couple[0] + coin_couple[1] + '_' + period + '_' + method + '.png')
                                plt.show()
                                net_lst_new = net_df.net.tolist()
                                ann_ROR = annROR(net_lst_new, n)
                                total_ret = net_lst_new[-1]
                                max_retrace_lst = [net_lst[i] for i in range(1, len(net_lst), int(1440 / n))]
                                max_retrace = maxRetrace(max_retrace_lst)
                                sharp = yearsharpRatio(net_lst_new, n)

                                signal_state = pd.DataFrame(signal_lst,
                                                            columns=['s_time', 'e_time', 'b_price', 's_price', 'ret',
                                                                     'max_ret', 'hold_day']) \
                                    .assign(method=method)
                                signal_state.to_csv('cl_trend/data/signal_tqa_' + str(n) + '.csv')
                                # df_lst.append(signal_state)
                                win_r, odds, ave_r, mid_r = get_winR_odds(signal_state.ret.tolist())
                                win_R_3, win_R_5, ave_max = get_winR_max(signal_state.max_ret.tolist())
                                state_row = []
                                state_row.append(coin_couple[0] + coin_couple[1])
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
                                state_row.append(N1)
                                state_row.append(N_ATR)
                                state_row.append(ATR_n)
                                state_row.append(road_period)
                                state_lst.append(state_row)
                                print('胜率=', win_r)
                                print('盈亏比=', odds)
                                print('总收益=', total_ret - 1)
                                print('年化收益=', ann_ROR)
                                print('夏普比率=', sharp)
                                print('最大回撤=', max_retrace)
                                print('交易次数=', len(signal_state))
                                print('平均每次收益=', ave_r)
                                print('平均持仓周期=', signal_state.hold_day.mean())
                                print('中位数收益=', mid_r)
                                print('超过3%胜率=', win_R_3)
                                print('超过5%胜率=', win_R_5)
                                print('平均最大收益=', ave_max)
                                print('参数=', method)

    signal_state = pd.DataFrame(state_lst, columns=['symble', 'tm', 'win_r', 'odds', 'total_ret', 'ann_ret', 'sharp', 'max_retrace',
                                                    'trade_times', 'ave_r', 'ave_hold_days', 'mid_r', 'win_r_3',
                                                    'win_r_5', 'ave_max', 'ma', 'art_N', 'art_n', 'tqa_td'])
    print(signal_state)
    # signal_state.to_csv('cl_trend/data/state_tqa_' + str(n) + '_' + str(x) + '_newstop.csv')
    # signal_df = pd.concat(df_lst)
    # signal_df.to_csv('cl_trend/data/signal_tqa_' + str(n) + '_' + str(x) + '_.csv')





