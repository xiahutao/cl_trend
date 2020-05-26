# coding=utf-8
'''
Created on 7.9, 2018

@author: fang.zhang
'''

from __future__ import division
import jqdatasdk
from jqdatasdk import *
import os
import time
jqdatasdk.auth('18610039264', 'zg19491001')


def stock_price(sec, period, sday, eday):
    """
    输入 股票代码，开始日期，截至日期
    输出 个股的前复权的开高低收价格
    """
    temp = get_price(sec, start_date=sday, end_date=eday, frequency=period,
                     fields=['open', 'close', 'low', 'high', 'volume', 'money', 'high_limit'], skip_paused=False,
                     fq='pre', count=None).assign(stock_code=sec).reset_index() \
        .rename(columns={'index': 'tradedate'})
    return temp


# 获取交易日
def tradeday(sday, eday):
    """
    输入 开始时间 和 截止时间
    输出 list 交易日 datetime格式
    """
    return get_trade_days(sday, eday)


if __name__ == '__main__':
    os.getcwd()
    os.chdir(r'/Users/zhangfang/PycharmProjects')
    print('a')
    t0 = time.time()
    data = opt.run_query(query(opt.OPT_CONTRACT_INFO).filter(opt.OPT_CONTRACT_INFO.underlying_symbol == '510050.XSHG'))
    print(data)
    data.to_csv('cl_trend/option/data/opt_info.csv', encoding='gbk')

