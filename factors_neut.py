#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 16:16:49 2019
基于USDT因子的BTC-USDT的中性策略
@author: lion95
"""

from __future__  import division
import pandas as pd
import numpy as np
from dataapi import *
import os
py_path=r'/Users/lion95/Documents/mywork/多因子'
os.chdir(py_path)
import talib as tb
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import math
import matplotlib.dates as mdates
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

alpha_test = ["Alpha.alpha047"]
if __name__=='__main__':

    symbols = ["btcusdt","ethusdt"]
    switch=False # True 越大越好  False 越小越好
    period='1d'
    exchange='BIAN'
    sday='2018-01-01'
    eday='2019-03-10'
    
#    N1_lst=np.arange(0.1,1,0.05) # 下阈值
#    N2_lst=np.arange(1.1,3,0.1)# 上阈值
#    
    N1_lst=[0.8] # 下阈值
    N2_lst=[1.3]# 上阈值    
    
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
        except (AttributeError, FileNotFoundError):
            print('is error')
        num=num+1 
        
    res.columns=['tradeDate','secID','factor']
    res=res.dropna() 
    res=res.pivot(index='tradeDate', columns='secID', values='factor')
    res['apha_ratio']=res.ethusdt/res.btcusdt
    res=res.reset_index()
    res.tradeDate=res.tradeDate.shift(-1)
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
    price_init_chg.columns=['btc_chg','eth_chg']
    
    df=res.merge(price_init_chg,on='tradeDate')  
    df=df.reset_index()

    df=df.dropna()    
    
    tradelist=df.tradeDate.tolist()
    
    result_lst=list()
    
    for N1 in N1_lst:
        for N2 in N2_lst:
            r_lst=list()
            e_lst=list()
            date_lst=list()
            method='neut_stratege_aph046'+'_'+str(N1)+'_'+str(N2)
            print(method)

            
            curve_lst=list()
            pos_lst=list()
            num=0
            trade_nums=0
            position=0 # 0:空仓 1：多ETH 空BTC 2：空ETH 多BTC 
            p_lst=list()
            for idx,row_ in df.iterrows():
        #        print(num%5)

         #       if (position==0) and (row_['GarmanKlassYang_Vol_1']>0.0) :
                if (position==0):
                    if (row_['apha_ratio'] < N1):
                        position=1
                        trade_nums=trade_nums+1
                    elif (row_['apha_ratio'] > N2):
                        position=2
                        trade_nums=trade_nums+1
                else:    
                    if (position==1) & (row_['apha_ratio'] > 1):
        #                print('************')
                        position=0
                    elif (position==2) & (row_['apha_ratio'] < 1):
                        position=0
                                    
                p_lst.append(position)                                    
                if position==0:
                    curve_lst.append(0)
                elif position in [1]:
                    curve_lst.append((row_['eth_chg']-row_['btc_chg'])/2)
        #            num=num+1
                elif position in [2]:
                    curve_lst.append((row_['btc_chg']-row_['eth_chg'])/2)
        #            num=num+1
                pos_lst.append(position)
            df=df.assign(chg_neu=curve_lst)\
                         .assign(position=pos_lst)\
                         .assign(curve_neu=lambda df:df.chg_neu+1)\
                         .assign(curve_neu=lambda df:df.curve_neu.cumprod())\
                         .assign(position=p_lst)
            df['tradeDate'] = pd.to_datetime(df['tradeDate'])
            fig, ax = plt.subplots(figsize=(20,10))
            Loc = mdates.DayLocator(interval=50)
            ax.xaxis.set_major_locator(Loc)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))  
        
            df[['tradeDate','curve_neu']].plot(x='tradeDate',ax=ax)   
#            df.to_csv('neu_stra_aph46.csv')     

                # 统计指标
            res_lst=df.curve_neu.tolist()
            r_lst.append((res_lst[-1]-res_lst[0])/res_lst[0])
            r_lst.append(annROR(res_lst))
            r_lst.append(maxRetrace(res_lst))
            r_lst.append(yearsharpRatio(res_lst))
            r_lst.append(trade_nums)
            if trade_nums>0:                    
                r_lst.append((res_lst[-1]-res_lst[0])/(res_lst[0]*trade_nums))
            else:
                r_lst.append(0)                    
            r_lst.append(method)
            result_lst.append(r_lst)
    res=pd.DataFrame(result_lst,columns=['ror','annror','maxRetrace','yearsharpRatio','trade_nums','avg_ror','method'])            
#    res.to_csv('neu_strategy_aph46.csv')      
