# -*- coding: utf-8 -*-
"""
Created on Sat Sep 09 15:14:26 2017

@author: Administrator
"""
# ==============================================================================
# 引入包   本代码计算单一股票唐奇安通道交易法 
# ==============================================================================
import pymssql
import pandas as pd
import os

os.getcwd()
os.chdir(u'E:\mywork\\fund')
import numpy as np
# import seaborn as sns
# import statsmodels.api as sm
# import numpy as np
import urllib  # 接口读数据的包
import datetime
import matplotlib.pyplot as plt


# ==============================================================================
# 定义函数
# ==============================================================================
def getQuote(code, sdate, edate):  # 提取个股数据
    code = str(code)
    sdate = str(sdate)
    edate = str(edate)
    Kbaseurl = 'http://fintech.jrj.com.cn/genius/get?'
    getQuery = ''
    getQuery = getQuery.join((Kbaseurl, 'code=', code, '&start=', sdate, '&end=', edate))
    #    pdb.set_trace()
    response = urllib.urlopen(getQuery)
    josndata = response.read()
    df_obj = pd.read_json(josndata)
    return (df_obj)


def H10(df, idx, HL):  # 计算每天前10日的最高价的最大值
    if idx < HL:
        return np.nan
    else:
        s0 = df.iloc[idx - HL:idx, :]
        res = s0.thigh.max()
        return res


def L10(df, idx, HL):  # 计算每天前10日的最低价的最小值
    if idx < HL:
        return np.nan
    else:
        s0 = df.iloc[idx - HL:idx, :]
        res = s0.tlow.min()
        return res


def M60(df, idx, Mnum):  # 计算每天60日线1
    if idx < Mnum:
        return np.nan
    else:
        s0 = df.iloc[idx - Mnum:idx, :]
        res = s0.tclose.mean()
        return res


def Truerange(df, idx):  # 计算真实波动率
    if idx < 1:
        return np.nan
    else:
        maxvalue = list()
        s0 = df.iloc[idx - 1:idx + 1, :]
        s0 = s0.set_index([range(len(s0))])
        maxvalue.append(s0.thigh[1] - s0.tlow[1])
        maxvalue.append(abs(s0.tclose[0] - s0.thigh[1]))
        maxvalue.append(abs(s0.tclose[0] - s0.tlow[1]))
        res = max(maxvalue)
        return res


def ATR(df, idx, ATRperiod):  # 计算真实波动率
    if idx < ATRperiod:
        return np.nan
    else:
        s0 = df.iloc[idx - ATRperiod:idx, :]
        res = s0.Truerange.mean()
        return res


# ==============================================================================
# 程序主体
# ==============================================================================
DATA_DIR = 'E:\mywork'
code = '399905.zs'
name = u'中证500'
HL = 25  # 轨道宽度
Mnum = 20  # 均线参数
# N=0.15# 止损比例
ATRperiod = 11  # ATR周期
ATR_N = 2.5  # ATR倍数

# HL=5    #轨道宽度
# Mnum=25 #均线参数
##N=0.15# 止损比例
# ATRperiod=10#ATR周期
# ATR_N=2#ATR倍数

# 获取数据，计算交易信号
jk = getQuote(code, '2013-01-01', '2017-09-10')  # 提取数据 并清洗
df = jk[['tradedate', 'tclose', 'thigh', 'tlow', 'topen']]
f = lambda s: s[:10]
df.tradedate = df.tradedate.astype(str)
df.tradedate = df.tradedate.apply(f)
# df=df[df['trade_status']==u'正常上市']
df0 = df.set_index([range(len(df))])  # 计算B,S 信号
df1 = df0.assign(H10=lambda df: [H10(df, idx, HL) for idx, row in df.iterrows()]) \
    .assign(L10=lambda df: [L10(df, idx, HL) for idx, row in df.iterrows()]) \
    .assign(M60=lambda df: [M60(df, idx, Mnum) for idx, row in df.iterrows()]) \
    .assign(Truerange=lambda df: [Truerange(df, idx) for idx, row in df.iterrows()]) \
    .assign(ATR=lambda df: [ATR(df, idx, ATRperiod) for idx, row in df.iterrows()]) \
    .assign(B=lambda df: df.thigh > df.H10) \
    .assign(S=lambda df: df.tlow < df.L10) \
    .assign(over60=lambda df: df.topen > df.M60) \
    .assign(up60=lambda df: df.M60 > df.M60.shift(1))

df1['TB'] = df1['B'].shift(1)
df1['TS'] = df1['S'].shift(1)
f1 = lambda s: 1 if s == True  else 0
df1.TB = df1.TB.apply(f1)
df1.TS = df1.TS.apply(f1)
df1.dropna(subset=['M60', 'ATR'], inplace=True)
df1 = df1.set_index([range(len(df1))])
# ==============================================================================
# 回测
# ==============================================================================
datelist = list(df1.tradedate)
flag = 0  # 0:没有持仓 1：持仓
flag1 = 0  # 0:表示正常开仓状态，1：处于止损状态
capital = 1000000
cash = 1000000
value = 0
TCO = 0
TCO1 = 0  # 入场后的最高价配合创新高
num = 0
win = 0
temp = pd.Series([])  # 持仓列表
temp1 = pd.Series([])  # 权益列表
temp2 = pd.Series([])  # 买卖明细列表
buyed = pd.DataFrame([[0, 0, 0, 0, 0]], columns=['TRADEDATE', 'STOCKCODE', 'STOCKSNAME', 'Price', 'LOTS'])
account = pd.DataFrame([[0, 0, 0, 0, 0]], columns=['DATE', 'CASH', 'VALUE', 'CAPITAL', 'POSITION'])
# detail=pd.DataFrame([[0,0,0,0,0,0]],columns=['DIRE','DATE','CODE','NAME','PRICE','LOTS'])
# 回测主体
for i in datelist:
    #    i=datelist[0]
    cl = df1[df1['tradedate'] == i]
    condS = (TCO > 0) & (cl['topen'].values[0] < TCO - ATR_N * cl['ATR'].values[0])
    if (flag == 0) & (cl['TB'].values[0] == 1) & (cl['over60'].values[0] == True) & (cl['up60'].values[0] == True):
        #        buyed=pd.DataFrame([[0,0,0,0,0]],columns=['TRADEDATE','STOCKCODE','STOCKSNAME','Price','LOTS'])
        buy_cost = cl.topen.values[0]
        buy_amount = ((cash // buy_cost) // 100) * 100
        buyed.iloc[0, 0] = i
        buyed.iloc[0, 1] = code
        buyed.iloc[0, 2] = name
        buyed.iloc[0, 3] = buy_cost
        buyed.iloc[0, 4] = buy_amount
        #        temp=pd.concat([temp,buyed])
        detail = pd.DataFrame([[0, 0, 0, 0, 0, 0]], columns=['DIRE', 'DATE', 'CODE', 'NAME', 'PRICE', 'LOTS'])
        detail.iloc[0, 0] = 'B'
        detail.iloc[0, 1] = i
        detail.iloc[0, 2] = code
        detail.iloc[0, 3] = name
        detail.iloc[0, 4] = buy_cost
        detail.iloc[0, 5] = buy_amount
        temp2 = pd.concat([temp2, detail])
        TCO = buy_cost
        TCO1 = 0
        cash = cash - buy_cost * buy_amount
        num = num + 1
        #        value=value+buy_cost*buy_amount
        flag = 1
        flag1 = 0
    if (flag == 1) & (condS | (cl['topen'].values[0] > 1.2 * buyed.iloc[
        0, 3])):  # 止盈止损条件| (cl['topen'].values[0]>buyed.iloc[0,3]+2*ATR_N*cl['ATR'].values[0])
        sell_price = cl.topen.values[0]
        if buyed.iloc[0, 3] > sell_price:
            win = win + 1
        sell_amount = buyed.LOTS.values[0]
        buyed.iloc[0, 0] = 0
        buyed.iloc[0, 1] = 0
        buyed.iloc[0, 2] = 0
        buyed.iloc[0, 3] = 0
        buyed.iloc[0, 4] = 0
        detail = pd.DataFrame([[0, 0, 0, 0, 0, 0]], columns=['DIRE', 'DATE', 'CODE', 'NAME', 'PRICE', 'LOTS'])
        detail.iloc[0, 0] = 'S'
        detail.iloc[0, 1] = i
        detail.iloc[0, 2] = code
        detail.iloc[0, 3] = name
        detail.iloc[0, 4] = sell_price
        detail.iloc[0, 5] = sell_amount
        temp2 = pd.concat([temp2, detail])
        cash = cash + sell_price * sell_amount * (1 - 0.003)
        #        value=0
        flag = 0
        TCO1 = TCO
        TCO = 0
        flag1 = 1
        if cl['topen'].values[0] > 1.2 * buyed.iloc[0, 3]:
            flag1 = 0
    if (flag == 0) & (flag1 == 1) & (cl['tclose'].values[
                                         0] > TCO1):  # 止盈止损条件 | (cl['topen'].values[0]>1.2*buyed.iloc[0,3]) | (cl['topen'].values[0]>buyed.iloc[0,3]+2*ATR_N*cl['ATR'].values[0])
        buy_cost = cl.topen.values[0]
        buy_amount = ((cash // buy_cost) // 100) * 100
        buyed.iloc[0, 0] = i
        buyed.iat[0, 1] = code
        buyed.iloc[0, 2] = name
        buyed.iloc[0, 3] = buy_cost
        buyed.iloc[0, 4] = buy_amount
        #        temp=pd.concat([temp,buyed])
        detail = pd.DataFrame([[0, 0, 0, 0, 0, 0]], columns=['DIRE', 'DATE', 'CODE', 'NAME', 'PRICE', 'LOTS'])
        detail.iloc[0, 0] = 'B'
        detail.iloc[0, 1] = i
        detail.iloc[0, 2] = code
        detail.iloc[0, 3] = name
        detail.iloc[0, 4] = buy_cost
        detail.iloc[0, 5] = buy_amount
        temp2 = pd.concat([temp2, detail])
        TCO = buy_cost
        TCO1 = 0
        cash = cash - buy_cost * buy_amount
        num = num + 1
        #        value=value+buy_cost*buy_amount
        flag = 1
        flag1 = 0
    buyed.iloc[0, 0] = i
    temp = pd.concat([temp, buyed])
    if (cl['thigh'].values[0] > TCO) & (TCO > 0):
        TCO = cl['thigh'].values[0]
    value = buyed.iloc[0, 4] * cl['tclose'].values[0]
    capital = value + cash
    account.iloc[0, 0] = i
    account.iloc[0, 1] = cash
    account.iloc[0, 2] = value
    account.iloc[0, 3] = capital
    account.iloc[0, 4] = value / capital
    temp1 = pd.concat([temp1, account])
# ==============================================================================
# 回测评价
# ==============================================================================
temp1 = temp1.iloc[:, 1:]
temp1.index = temp1.DATE

plt.subplot(211)
(temp1.CAPITAL / 1000000).plot()
plt.title('curve of value')
plt.subplot(212)
temp1.POSITION.plot()
plt.title('position')
nvalue1 = (temp1.CAPITAL / 1000000)
start = '2016-01-01'
end = '2017-08-31'
# 交易时间
print('交易区间为：%s--%s' % (start, end))
# 交易次数统计,胜率统计
print('交易次数%d' % num + '次')
print('盈利次数%d' % win + '次')
winratio = round(float(win) * 100 / num, 2)
print('交易胜率为%0.2f' % winratio + '%')
# 总收益率
gy = (nvalue1[-1] - 1) * 100
print('总收益率：%0.2f' % gy + '%')
# 年化收益率
jkdt = pd.to_datetime(jk['tradedate'])
t = list(jkdt)
td = round(float((t[-1] - t[0]).days) / 365, 2)
ay = round(((nvalue1[-1] ** (1.0 / td)) - 1) * 100, 2)
print('年化收益率：%0.2f' % ay + '%')
MDD = 0  # 最大回撤
MAX = nvalue1[0]
for i in range(len(nvalue1)):
    if nvalue1[i] > MAX:
        MAX = nvalue1[i]
    MDD1 = (MAX - nvalue1[i]) / MAX
    if MDD1 > MDD:
        MDD = MDD1
MDD = round(MDD * 100, 2)
print('最大回撤比率：%0.2f' % MDD + '%')
# 风险收益比
risktoget = ay / MDD
risktoget = round(risktoget, 2)
print('风险收益比：%0.2f' % risktoget)
# 计算夏普比率
ndv = []  # 日收益率
account1 = list(temp1.CAPITAL)
for i in range(len(account1)):
    if i == 0:
        continue
    else:
        ndv.append((account1[i] - account1[i - 1]) / account1[i - 1])
ytemp_np = np.array(ndv)
std = np.std(ytemp_np, ddof=1)
std1 = std * (252 ** (0.5))
sharp = (ay / 100 - 0.024) / std1  # 全局夏普
print('夏普比率为：%0.2f' % sharp)
# print([HL,Mnum,N])
# temp.to_csv(os.path.join(DATA_DIR, "shiyan.csv"),encoding='gbk')
# temp1.to_csv(os.path.join(DATA_DIR, "shiyan2.csv"),encoding='gbk')
# temp2.to_csv(os.path.join(DATA_DIR, "shiyan3.csv"),encoding='gbk')
temp1.to_csv("'{name1}'.csv".format(name1=code), encoding='gbk')
# ==============================================================================
# code='399905.zs'
# name=u'中证500'

# HL=25     #轨道宽度
# Mnum=20 #均线参数
##N=0.15# 止损比例
# ATRperiod=11#ATR周期
# ATR_N=2.5#ATR倍数

# ==============================================================================
# df1.to_csv('1.csv',encoding='gbk')


# ==============================================================================
## code='000016.zs'
# name=u'上证50'
##HL=11       #轨道宽度
##Mnum=17 #均线参数
###N=0.15# 止损比例
##ATRperiod=25#ATR周期
##ATR_N=2#ATR倍数
#
# HL=5    #轨道宽度
# Mnum=25 #均线参数
##N=0.15# 止损比例
# ATRperiod=10#ATR周期
# ATR_N=2#ATR倍数
# ==============================================================================
print([HL, Mnum, ATRperiod, ATR_N])
