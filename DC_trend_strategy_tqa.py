# -*- coding: utf-8 -*-
"""
Created on Sun Jul 08 11:01:46 2018

@author: Administrator
"""
# =============================================================================
# 引包 基于创新高的唐其安通道策略
# =============================================================================
import pandas as pd
import os

os.getcwd()
os.chdir(u'D:\mywork\DC')
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# 函数
# =============================================================================
# def getQuote(code, sdate, edate):#提取个股数据
#    code = str(code)
#    sdate = str(sdate)
#    edate = str(edate)
#    Kbaseurl = 'http://fintech.jrj.com.cn/genius/get?'
#    getQuery = ''
#    getQuery = getQuery.join((Kbaseurl, 'code=', code, '&start=', sdate, '&end=', edate))
##    pdb.set_trace()
#    response = urllib.urlopen(getQuery)
#    josndata = response.read()
#    df_obj = pd.read_json(josndata)
#    return (df_obj)

def H10(df, idx, HL):  # 计算每天前HL日的最高价的最大值
    if idx < HL:
        return np.nan
    else:
        s0 = df.iloc[idx - HL:idx, :]
        res = s0.fac_thigh.max()
        return res


def L10(df, idx, HL):  # 计算每天前HL日的最低价的最小值
    if idx < HL:
        return np.nan
    else:
        s0 = df.iloc[idx - HL:idx, :]
        res = s0.fac_tlow.min()
        return res


def M60(df, idx, Mnum):  # 计算Mnum日均线
    if idx < (Mnum - 1):
        return np.nan
    else:
        s0 = df.iloc[idx - (Mnum - 1):idx + 1, :]
        res = s0.fac_tclose.mean()
        return res


def Truerange(df, idx):  # 计算真实波动率
    if idx < 1:
        return np.nan
    else:
        maxvalue = list()
        s0 = df.iloc[idx - 1:idx + 1, :]
        s0 = s0.set_index([range(len(s0))])
        maxvalue.append(s0.fac_thigh[1] - s0.fac_tlow[1])
        maxvalue.append(abs(s0.fac_tclose[0] - s0.fac_thigh[1]))
        maxvalue.append(abs(s0.fac_tclose[0] - s0.fac_tlow[1]))
        res = max(maxvalue)
        return res


def ATR(df, idx, ATRperiod):  # 计算ATR值
    if idx < ATRperiod:
        return np.nan
    else:
        s0 = df.iloc[idx - ATRperiod:idx, :]
        res = s0.Truerange.mean()
        return res


# =============================================================================
# 读取数据
# =============================================================================
if __name__ == '__main__':
    df = pd.read_csv('btc_usd_day.csv')
    df_big = df.query("date>'2016/1/1'")

    HL = 5  # 轨道宽度
    Mnum = 20  # 均线参数
    # N=0.15# 止损比例
    ATRperiod = 5  # ATR周期
    ATR_N = 1  # ATR倍数

    df = df_big.copy()  # 提取数据 并清洗
    df.columns = ['tradedate', 'fac_topen', 'fac_thigh', 'fac_tlow', 'fac_tclose', 'vol']
    # 生成唐奇安通道以及均线，买入信号，卖出信号以及均线状态
    df0 = df.set_index([range(len(df))])  # 计算B,S 信号
    df1 = df0.assign(H10=lambda df: [H10(df, idx, HL) for idx, row in df.iterrows()]) \
        .assign(L10=lambda df: [L10(df, idx, HL) for idx, row in df.iterrows()]) \
        .assign(M60=lambda df: [M60(df, idx, Mnum) for idx, row in df.iterrows()]) \
        .assign(Truerange=lambda df: [Truerange(df, idx) for idx, row in df.iterrows()]) \
        .assign(ATR=lambda df: [ATR(df, idx, ATRperiod) for idx, row in df.iterrows()]) \
        .assign(B=lambda df: df.fac_thigh > df.H10) \
        .assign(S=lambda df: df.fac_tlow < df.L10) \
        .assign(over60=lambda df: df.fac_topen > df.M60) \
        .assign(up60=lambda df: df.M60 > df.M60.shift(1))

    df1['TB'] = df1['B'].shift(1)
    df1['TS'] = df1['S'].shift(1)
    f1 = lambda s: 1 if s == True else 0
    df1.TB = df1.TB.apply(f1)
    df1.TS = df1.TS.apply(f1)
    df1.dropna(subset=['M60', 'ATR'], inplace=True)
    df1 = df1.set_index([range(len(df1))])
    datelist = list(df1.tradedate)
    flag = 0  # 0:没有持仓 1：持仓
    capital = 1000000
    cash = 1000000
    value = 0
    TCO = 0  # 入场后的最高价
    TCO1 = 0  # 止损前的最高价，配合创新高
    num = 0
    win = 0
    flag = 0  # 0:没有持仓 1：持仓
    flag1 = 0  # 0:表示正常开仓状态，1：处于止损状态

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
        condS = (TCO > 0) & (cl['fac_topen'].values[0] < TCO - ATR_N * cl['ATR'].values[0])
        if (flag == 0) & (cl['TB'].values[0] == 1) & (cl['over60'].values[0] == True) & (cl['up60'].values[0] == True):
            #        buyed=pd.DataFrame([[0,0,0,0,0]],columns=['TRADEDATE','STOCKCODE','STOCKSNAME','Price','LOTS'])
            buy_cost = cl.fac_topen.values[0]
            buy_amount = ((cash // buy_cost) // 100) * 100
            buyed.iloc[0, 0] = i
            buyed.iloc[0, 1] = 'BTC'
            buyed.iloc[0, 2] = 'BTC'
            buyed.iloc[0, 3] = buy_cost
            buyed.iloc[0, 4] = buy_amount
            #        temp=pd.concat([temp,buyed])
            detail = pd.DataFrame([[0, 0, 0, 0, 0, 0]], columns=['DIRE', 'DATE', 'CODE', 'NAME', 'PRICE', 'LOTS'])
            detail.iloc[0, 0] = 'B'
            detail.iloc[0, 1] = i
            detail.iloc[0, 2] = 'BTC'
            detail.iloc[0, 3] = 'BTC'
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
        if (flag == 1) & (condS | (cl['fac_topen'].values[0] > buyed.iloc[0, 3] + 2 * ATR_N * cl['ATR'].values[
            0])):  # 止盈止损条件| (cl['fac_topen'].values[0]>buyed.iloc[0,3]+2*ATR_N*cl['ATR'].values[0])
            sell_price = cl.fac_topen.values[0]
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
            detail.iloc[0, 2] = 'BTC'
            detail.iloc[0, 3] = 'BTC'
            detail.iloc[0, 4] = sell_price
            detail.iloc[0, 5] = sell_amount
            temp2 = pd.concat([temp2, detail])
            cash = cash + sell_price * sell_amount * (1 - 0.003)
            #        value=0
            flag = 0
            TCO1 = TCO
            TCO = 0
            flag1 = 1
        if (flag == 0) & (flag1 == 1) & (cl['fac_tclose'].values[
                                             0] > TCO1):  # 止盈止损条件 | (cl['fac_topen'].values[0]>1.2*buyed.iloc[0,3]) | (cl['fac_topen'].values[0]>buyed.iloc[0,3]+2*ATR_N*cl['ATR'].values[0])
            buy_cost = cl.fac_topen.values[0]
            buy_amount = ((cash // buy_cost) // 100) * 100
            buyed.iloc[0, 0] = i
            buyed.iloc[0, 1] = 'BTC'
            buyed.iloc[0, 2] = 'BTC'
            buyed.iloc[0, 3] = buy_cost
            buyed.iloc[0, 4] = buy_amount
            #        temp=pd.concat([temp,buyed])
            detail = pd.DataFrame([[0, 0, 0, 0, 0, 0]], columns=['DIRE', 'DATE', 'CODE', 'NAME', 'PRICE', 'LOTS'])
            detail.iloc[0, 0] = 'B'
            detail.iloc[0, 1] = i
            detail.iloc[0, 2] = 'BTC'
            detail.iloc[0, 3] = 'BTC'
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
        if (cl['fac_thigh'].values[0] > TCO) & (TCO > 0):
            TCO = cl['fac_thigh'].values[0]
        value = buyed.iloc[0, 4] * cl['fac_tclose'].values[0]
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
    start = temp1.DATE.tolist()[0]
    end = temp1.DATE.tolist()[-1]
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
    jkdt = pd.to_datetime(temp1.DATE)
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

    temp.to_csv(os.path.join("shiyan.csv"), encoding='gbk')
    temp1.to_csv(os.path.join("shiyan2.csv"), encoding='gbk')
    temp2.to_csv(os.path.join("shiyan3.csv"), encoding='gbk')
#  df_big.close.plot()
