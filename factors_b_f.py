#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:17:08 2019
基于96号因子计算  进行单因子币对优选 增加
均线的过滤器
@author: lion95
"""

from __future__ import division
import sys
sys.path.append('..')
#from lib.myfun import *
#from lib.factors_gtja import *
#from lib.dataapi import *
import numpy as np
import pandas as pd
import copy
import os
py_path=r'/Users/lion95/Documents/mywork/多因子'
os.chdir(py_path)
import datetime
import math
import talib as tb
from dataapi import *
from factors_gtja import *

def yearsharpRatio(netlist, n=240):
    row = []
    for i in range(1, len(netlist)):
        row.append(math.log(netlist[i] / netlist[i - 1]))
    return np.mean(row) / np.std(row) * math.pow(365 * 1440/n, 0.5)

def maxRetrace(list):
    '''
    :param list:netlist
    :return: 最大历史回撤
    '''
    Max = 0
    for i in range(len(list)):
        if 1 - list[i] / max(list[:i + 1]) > Max:
            Max = 1 - list[i] / max(list[:i + 1])

    return Max


def annROR(netlist,n=240):
    '''
    :param netlist:净值曲线
    :return: 年化收益
    '''
    return math.pow(netlist[-1] / netlist[0], 365 * 1440 / len(netlist) / n) - 1





def price_volume(df, corr_window):
    # 计算因子：量价相关系数
    factor = talib.CORREL(df['close'], df['volume'], corr_window)
    return factor


def build_col_name(factor_name, param):
    col_name = str(factor_name)
    for k, v in param.items():
        col_name += '_' + str(v)
    return col_name


alpha_test = ["Alpha.alpha096"]

if __name__ == '__main__':

    # exchange = 'BITFINEX'
#    symbols = ["ethbtc", "xrpbtc", "eosbtc", "tusdbtc", "ltcbtc",
#           "trxbtc", "adabtc", "bnbbtc",'neobtc','xlmbtc',
#           "etcbtc","ontbtc","bchabcbtc", "bchsvbtc"]
    symbols = ["ethbtc", "xrpbtc", "eosbtc", "tusdbtc", "ltcbtc",
           "trxbtc", "adabtc", "bnbbtc",'neobtc','xlmbtc',
           "etcbtc","ontbtc","bchabcbtc", "bchsvbtc"]    
    switch=True # True 越大越好  False 越小越好
    period='4h'
    exchange='BIAN'
    sday='2018-01-01'
    eday='2019-03-10'

    num=0
    for symbol in symbols:
        try:
            if num==0:
                dataf = pd.read_csv('data/'+symbol+'_'+period+'_'+exchange+'_'+sday+'_'+eday+'.csv', index_col=0)
                print(symbol)
                Alpha = Alphas(dataf)
                col_name = alpha_test[0]
                df_m = copy.deepcopy(dataf)
                df_m=df_m[['date']]
                df_m['kind']=symbol            
                df_m[col_name] = eval(alpha_test[0])()
                res=df_m
                df_f=dataf[['date','close']]
                df_f=df_f.assign(ma5=lambda df:tb.MA(df['close'].values,15,matype=1))
                df_f['kind']=symbol
                df_f['uma5']=df_f['ma5']-df_f['close']
                df_f=df_f[['date','kind','uma5']]
#                df_f=df_f.rename(columns={'uma5':symbol+'_uma5'})
                res_f=df_f.copy()
                
            else:
                dataf = pd.read_csv('data/'+symbol+'_'+period+'_'+exchange+'_'+sday+'_'+eday+'.csv', index_col=0)
                print(symbol)
                Alpha = Alphas(dataf)
                col_name = alpha_test[0]
                df_m = copy.deepcopy(dataf)
                df_m=df_m[['date']]
                df_m['kind']=symbol            
                df_m[col_name] = eval(alpha_test[0])()
                res=pd.concat([res,df_m])
                
                df_f=dataf[['date','close']]
                df_f=df_f.assign(ma5=lambda df:tb.MA(df['close'].values,15,matype=1))
                df_f['kind']=symbol
                df_f['uma5']=df_f['ma5']-df_f['close']
                df_f=df_f[['date','kind','uma5']]
#               df_f=df_f.rename(columns={'uma5':symbol+'_uma5'}) 
                res_f=pd.concat([res_f,df_f]) 
        except (AttributeError, FileNotFoundError):
            print('is error')
        num=num+1 
    
    res.columns=['tradeDate','secID','factor']
    res=res.dropna()    

    res_f.columns=['tradeDate','secID','uma']
    res_f=res_f.dropna()
    
    res=res.merge(res_f,on=['tradeDate','secID'])

    trades=list()
    for idx,group_ in res.groupby('tradeDate'):
        detail=list()
        temp=group_.copy()
        temp=temp.sort_values(by='factor',ascending=switch)
        kinds=temp.secID.tolist()
        uma_lst=temp.uma.tolist()
        detail.append(idx)
        detail.append(kinds[-1])
        detail.append(uma_lst[-1])
        trades.append(detail)
    trade_df=pd.DataFrame(trades)
    trade_df.columns=['tradeDate','B','uma']
    trade_df.tradeDate=trade_df.tradeDate.shift(-1)
    trade_df=trade_df.dropna()
#    res=res.pivot(index='tradeDate', columns='secID', values='factor')
# 加载行情数据
    num=0
    for sy in symbols:
#        data_file=sy+'_4h_BIAN.csv'
#        print(data_file)
        if num==0:
            price=pd.read_csv('data/'+sy+'_'+period+'_'+exchange+'_'+sday+'_'+eday+'.csv', index_col=0)
            price=price[['date','close']]
            price['kind']=sy
        else:
            temp=pd.read_csv('data/'+sy+'_'+period+'_'+exchange+'_'+sday+'_'+eday+'.csv', index_col=0)
            temp=temp[['date','close']]
            temp['kind']=sy
            price=pd.concat([price,temp])
        num=num+1   
    price.columns=['tradeDate','closePrice','secID']
    price_init=price[['tradeDate','secID','closePrice']]
    price_init=price_init.pivot(index='tradeDate', columns='secID', values='closePrice')
    price_init=price_init.fillna(method='bfill')
    price_init_chg=(price_init-price_init.shift(1))/price_init.shift(1)
    
    
    
   # 回测
    df=trade_df.merge(price_init_chg,on='tradeDate')
    df.tradeDate=pd.to_datetime(df.tradeDate)
    df=df.set_index(df.tradeDate)
    curvechg=list()
    num=0
    for idx,row in df.iterrows():
        rec=list()
        if (num%3==0):
            if (row['uma']<0):
                b=row['B']
                c=2/1000
            else:
                b='btc'
                c=2/1000
        if b=='btc':
            buy=0
            rec.append(idx)
            rec.append(b)
            rec.append(buy-c)
            curvechg.append(rec)
            c=0
        else:
            buy=row[b]
            rec.append(idx)
            rec.append(b)
            rec.append(buy-c)
            curvechg.append(rec)
            c=0            
        num=num+1
    


    curve=pd.DataFrame(curvechg,columns=['date','kind','chg'])
    curve=curve.assign(oneadd=lambda df:1+df.chg)\
               .assign(f_curve=lambda df:df.oneadd.cumprod())\
               .assign(d_curve_f=lambda df:df.chg.cumsum())\
               .assign(d_curve=lambda df:1+df.d_curve_f)
    curve.to_excel('trade_lst.xls')
    curve.plot(x='date',y='d_curve',label='096')
    cv=curve.d_curve.tolist()
    print([annROR(cv),maxRetrace(cv),yearsharpRatio(cv)])
    
    
    