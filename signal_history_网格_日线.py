# coding=utf-8
'''
Created on 7.9, 2018

@author: fang.zhang
'''
from __future__ import division
from backtest_func import *
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')


def get_band(x, position):
    '''

    :param x: 网格宽度初始值
    :param position: 仓位（0-1）
    :return: 网格下线宽度
    '''
    if position <= 0.1:
        band_up = 5 * x
        band_down = x
    elif position <= 0.4:
        band_up = (4 * np.sin(5/4 * np.pi * (1 - position) - 9/8 * np.pi) + 5) * x
        band_down = x
    elif position < 0.6:
        band_up = x
        band_down = x
    elif position <= 0.9:
        band_up = x
        band_down = (4 * np.sin(5 / 4 * np.pi * position - 9 / 8 * np.pi) + 5) * x
    else:
        band_up = x
        band_down = 5 * x
    return band_up, band_down


if __name__ == '__main__':
    os.getcwd()
    print(os.path)
    os.chdir(r'/Users/zhangfang/PycharmProjects')
    t0 = time.time()
    # symble = 'yeeeth'
    x = 4
    symble_lst = ['ethbtc', 'htbtc', 'bchbtc', 'etcbtc', 'xrpbtc', 'iotabtc', 'eosbtc', 'iostbtc', 'btcusdt']
    n = 1
    # N1_lst = [20]  # 均线周期
    # P_ATR_LST = [15]  # ATR周期
    # ATR_n_lst = [1]  # ATR倍数
    # road_period_lst = [10]  # 唐奇安通道周期

    P_ATR_LST = [20]  # ATR周期
    band_lst = [0.01, 0.008]  # 网格初始宽度
    ATR_n_lst = [1]  # ATR倍数
    weight_lst = [0.005, 0.004, 0.0025]  # 每次补仓权重

    fee = 0.002
    # s_date = '2016-01-01'
    # e_date = '2018-07-14'
    # group.loc[:, ['high', 'low', 'close', 'open']] = group.loc[:, ['high', 'low', 'close', 'open']]\
    #     .apply(lambda x: 1/x)
    # group = group.rename(columns={'high': 'low', 'low': 'high'})
    # print(group)
    df_lst = []
    lst = []
    state_lst = []
    for symble in symble_lst:
        group = pd.read_csv('huobiservice/data/' + symble + '_' + str(n) + 'min.csv').reset_index()
        print(group)
        group_day = pd.read_csv('huobiservice/data/' + symble + '_1440min.csv').reset_index()[
            ['date_time', 'high', 'low', 'close']]\
            .assign(date_time=lambda df: df.date_time.apply(lambda x: x + ' 00:00:00'))

        for N_ATR in P_ATR_LST:
            group_day['atr'] = talib.ATR(group_day['high'].values, group_day['low'].values,
                                         group_day['close'].values, N_ATR)
            print(group_day)
            group_day['top'], group_day['mid'], group_day['bottom'] = talib.BBANDS(group_day['close'].values,
                                                                                   timeperiod=20,
                                                                                   nbdevup=2, nbdevdn=2, matype=0)
            day_atr = group_day[['date_time', 'atr', 'bottom']] \
                .assign(atr=lambda df: df.atr.shift(1)) \
                .merge(group, on=['date_time'], how='right') \
                .sort_values(['date_time']).fillna(method='ffill').drop(['index'], axis=1)\
                .assign(close_1=lambda df: df.close.shift(1))\
                .assign(bottom=lambda df: df.bottom.shift(1))
            print(day_atr)
            day_atr = day_atr.dropna()
            print(day_atr)
            # day_atr.to_csv('huobiservice/data/day_atr.csv')
            for ATR_n in ATR_n_lst:
                for band in band_lst:

                    for weight in weight_lst:

                        signal_lst = []
                        trad_times = 0

                        if len(group) > 20:
                            net = 1
                            net_lst = []

                            position = 0
                            position_lst = []
                            position_lst.append(position)
                            high_price_pre = 1000000
                            base_price = day_atr.close_1.tolist()[0]
                            # signal_row = []
                            # stock_row = []
                            for idx, _row in day_atr.iterrows():
                                band_up, band_low = get_band(band, position)
                                if (position < weight) & (_row.low < _row.bottom):
                                    position = 0.3
                                    base_price = _row.bottom
                                    net = net * (0.7 + 0.3 * _row.close / _row.bottom)
                                    net_lst.append(net)
                                    position_lst.append(position)
                                    signal_row = []
                                    signal_row.append(_row.date_time)
                                    signal_row.append(_row.bottom)
                                    signal_row.append('b')
                                    signal_row.append(position)
                                    signal_lst.append(signal_row)

                                elif (position >= weight) & (_row.high >= base_price * (1 + band_up)):
                                    trad_times += 1
                                    position = position - weight
                                    s_time = _row.date_time
                                    s_price = base_price * (1 + band_up)
                                    net = net * ((1 - position - weight) + position * (_row.close / _row.close_1) +
                                                 weight * (s_price / _row.close_1) * (1 - fee))
                                    base_price = s_price
                                    signal_row = []
                                    signal_row.append(_row.date_time)
                                    signal_row.append(s_price)
                                    signal_row.append('s')
                                    signal_row.append(position)
                                    signal_lst.append(signal_row)

                                elif (position <= 1 - weight) & (_row.low <= base_price * (1 - band_low)):
                                    trad_times += 1
                                    position = position + weight
                                    s_time = _row.date_time
                                    cost = base_price * (1 - band_low)
                                    net = net * ((1 - position) + (position - weight) * (_row.close / _row.close_1) +
                                                 weight * (_row.close / cost) * (1 - fee))
                                    base_price = cost
                                    signal_row = []
                                    signal_row.append(_row.date_time)
                                    signal_row.append(cost)
                                    signal_row.append('b')
                                    signal_row.append(position)
                                    signal_lst.append(signal_row)

                                else:
                                    net = net * ((1 - position) + position * (_row.close / _row.close_1))

                                net_lst.append(net)
                                position_lst.append(position)
                            all_lst = []
                            all_lst.append(day_atr.date_time.tolist())
                            all_lst.append(day_atr.close.tolist())
                            all_lst.append(net_lst)
                            all_lst.append(position_lst)
                            all_lst = map(list, zip(*all_lst))

                            net_df = pd.DataFrame(list(all_lst), columns=['date_time', 'close', 'net', 'position']) \
                                .assign(close=lambda df: df.close / df.close.tolist()[0])
                            # net_df.to_csv('cl_trend/data/net_wangge' + symble + '.csv')
                            net_df[['date_time', 'net', 'close', 'position']].plot(
                                x='date_time', kind='line', grid=True, secondary_y='position', title=symble + '_' + str(n))
                            plt.xlabel('date_time')
                            plt.savefig('cl_trend/data/' + symble + '_' + str(n) + '_' + '.png')
                            plt.show()
                            ann_ROR = annROR(net_lst, n)
                            print('ann_ror:', ann_ROR)
                            total_ret = net_lst[-1]
                            print('total', total_ret)
                            max_retrace_lst = [net_lst[i] for i in range(1, len(net_lst), int(1440/n))]
                            max_retrace = maxRetrace(max_retrace_lst)
                            # max_retrace = 0.1
                            print('max_retrace', max_retrace)
                            sharp = yearsharpRatio(net_lst, n)
                            print('sharp', sharp)

                            # signal_state = pd.DataFrame(signal_lst, columns=['date_time', 'price', 'flag', 'position'])
                            # print(signal_state)
                            # signal_state.to_csv('cl_trend/data/signal_wangge_' + symble + '_.csv')
                            # df_lst.append(signal_state)

                            state_row = []
                            state_row.append(symble)
                            state_row.append(n)

                            state_row.append(total_ret - 1)
                            state_row.append(ann_ROR)
                            state_row.append(sharp)
                            state_row.append(max_retrace)
                            state_row.append(len(signal_lst))
                            state_row.append(np.mean(position_lst))

                            state_row.append(band)
                            state_row.append(N_ATR)
                            state_row.append(ATR_n)
                            state_row.append(weight)
                            state_lst.append(state_row)
                            print('总收益=', total_ret - 1)
                            print('年化收益=', ann_ROR)
                            print('夏普比率=', sharp)
                            print('最大回撤=', max_retrace)
                            print('交易次数=', len(signal_lst))
                            print('参数=', band, N_ATR, ATR_n, weight)

    signal_state = pd.DataFrame(state_lst, columns=['symble', 'tm', 'total_ret', 'ann_ret', 'sharp', 'max_retrace',
                                                    'trade_times', 'position', 'band', 'art_N', 'art_n', 'weight'])
    print(signal_state)
    signal_state.to_csv('cl_trend/data/state_wangge_' + symble + '_' + str(n) + 'all.csv')
    # signal_df = pd.concat(df_lst)
    # signal_df.to_csv('cl_trend/data/signal_tqa_' + str(n) + '_' + str(x) + '_.csv')





