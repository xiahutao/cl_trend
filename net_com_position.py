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


if __name__ == '__main__':
    os.getcwd()
    print(os.path)
    n=1
    os.chdir(r'/Users/zhangfang/PycharmProjects')
    symble_lst = ['htbtc', 'eosbtc', 'etcbtc', 'bchbtc']
    df = pd.DataFrame(columns=['date_time'])
    df_positon = pd.DataFrame(columns=['date_time'])
    for symble in symble_lst:
        csv = pd.read_csv('cl_trend/data/net_wangge_' + symble + '_.csv')[['date_time', 'net']]
        csv_position = pd.read_csv('cl_trend/data/net_wangge_' + symble + '_.csv')[['date_time', 'position']]
        df = df.merge(csv, on=['date_time'], how='outer')
        df_positon = df_positon.merge(csv_position, on=['date_time'], how='outer')

    df = df.sort_values(['date_time'])
    print(df)
    df_positon = df_positon.sort_values(['date_time'])
    print(df_positon)
    df = df.fillna(method='ffill').fillna(value=1).set_index(['date_time'], drop=True)
    print(df)
    df_positon = df_positon.fillna(method='ffill').fillna(value=0).set_index(['date_time'], drop=True)
    print(df_positon)
    df['net_n'] = df.mean(axis=1)
    df_positon['position_n'] = df_positon.mean(axis=1)
    print(df)
    df = df.reset_index(drop=False).merge(df_positon.reset_index(drop=False), on=['date_time'])
    df[['date_time', 'net_n', 'position_n']].plot(
                                x='date_time', kind='line', grid=True, title=str(n), secondary_y='position_n')
    plt.xlabel('date_time')
    plt.savefig('cl_trend/data/' + '组合净值' + str(n) + '.png')
    plt.show()
    df.to_csv('cl_trend/data/net_wangge_' + str(n) + '.csv')
    net_lst_new = df.net_n.tolist()

    ann_ROR = annROR(net_lst_new, n)
    total_ret = net_lst_new[-1]
    max_retrace_lst = [net_lst_new[i] for i in range(1, len(net_lst_new), int(1440 / n))]
    max_retrace = maxRetrace(max_retrace_lst)
    sharp = yearsharpRatio(net_lst_new, n)
    print('回测开始日期=', df.date_time.tolist()[0])
    print('回测结束日期=', df.date_time.tolist()[-1])
    print('总收益=', total_ret - 1)
    print('年化收益=', ann_ROR)
    print('夏普比率=', sharp)
    print('最大回撤=', max_retrace)