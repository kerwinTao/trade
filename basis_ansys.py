import pandas as pd
import tushare as ts
pro = ts.pro_api()
import pdb
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
import numpy as np
from tqdm import tqdm
import time
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def ansys_sigle(name, ts_code=None, industry=None, plot_on=True):

    while 1:
        try:
            if ts_code == None and industry == None:
                # 查询ts_code
                stocks_list = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
                stock = stocks_list[stocks_list['name'] == name]
                ts_code = stock['ts_code'].values[0]
                industry = stock['industry'].values[0]
            # 当前股价
            pice = ts.pro_bar(ts_code=ts_code)[['trade_date', 'close']]
            orig_data = pice
            orig_data['name'] = name

            # 总股本
            total_share = pro.daily_basic(ts_code=ts_code, fields='trade_date,total_share,pe, pb')
            orig_data = pd.merge(orig_data, total_share, how='outer', on='trade_date')

            # 商誉
            goodwill = pro.balancesheet(ts_code=ts_code,  fields='ann_date, goodwill')
            goodwill.rename(columns={'ann_date': 'trade_date'}, inplace=True)
            orig_data = pd.merge(orig_data, goodwill, how='outer', on='trade_date')

            # 基本每股收益, 每股净资产
            ps = pro.fina_indicator(ts_code=ts_code,  fields='ann_date, eps, bps')
            ps.rename(columns={'ann_date': 'trade_date'}, inplace=True)
            orig_data = pd.merge(orig_data, ps, how='outer', on='trade_date')
            break
        except:
            print('读取频率太高，等待10秒')
            time.sleep(10)

    # # 当前股价/基本每股收益
    # orig_data['指标1'] = orig_data['close'] / orig_data['eps']
    #
    # # 当前股价/每股净资产
    # orig_data['指标2'] = orig_data['close'] / orig_data['bps']

    # 按时间排序
    orig_data = orig_data.sort_values(by="trade_date", ascending=True)

    # 插值填充NaN
    orig_data = orig_data.interpolate()

    # 绘图
    if plot_on:
        fig = plt.figure()
        ax = fig.add_subplot(2, 2, 1)
        plt.plot(orig_data['trade_date'], orig_data['goodwill'], label='goodwill')
        ax.xaxis.set_major_locator(LinearLocator(numticks=8))
        plt.xticks(rotation=45)
        ax.legend()

        ax = fig.add_subplot(2, 2, 2)
        ax.plot(orig_data['trade_date'], orig_data['total_share'], label='total_share')
        ax.xaxis.set_major_locator(LinearLocator(numticks=8))
        plt.xticks(rotation=45)
        ax.legend()

        ax = fig.add_subplot(2, 2, 3)
        ax.plot(orig_data['trade_date'], orig_data['pe'], label='PE')
        ax.xaxis.set_major_locator(LinearLocator(numticks=8))
        plt.xticks(rotation=45)
        ax.legend()

        ax = fig.add_subplot(2, 2, 4)
        ax.plot(orig_data['trade_date'], orig_data['pb'], label='PB')
        ax.xaxis.set_major_locator(LinearLocator(numticks=8))
        plt.xticks(rotation=45)
        ax.legend()
        print(industry)
        plt.show()

    return industry, orig_data


def ansys_indus(industry, plot_on=True):
    stocks_list = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
    stocks_list = stocks_list[stocks_list['industry'] == industry]
    top10_PE = np.ones((10, 2)) * 1e99
    top10_PB = np.ones((10, 2)) * 1e99
    data_list_PE = [[]] * 10
    data_list_PB = [[]] * 10

    print('[-%s-]板块共计[-%d-]只股票，正在排序中...' % (industry, len(stocks_list)))
    for i, val in tqdm(stocks_list.iterrows()):
        _, data = ansys_sigle(name=val['name'], ts_code=val['ts_code'], industry=val['industry'], plot_on=False)
        pe, pb = data.iloc[-1, :][['pe', 'pb']]

        # 筛选
        if len(pe) > 0 and pe < top10_PE[-1, 0]:
            top10_PE[-1, :] = [pe, pb]
            data_list_PE[-1] = data

        if len(pe) > 0 and pb < top10_PB[-1, 1]:
            top10_PB[-1, :] = [pe, pb]
            data_list_PB[-1] = data

        # 排序
        data_list_PE = [data_list_PE[x] for x in top10_PE[:, 0].argsort()]
        data_list_PB = [data_list_PB[x] for x in top10_PB[:, 1].argsort()]
        top10_PE = top10_PE[top10_PE[:, 0].argsort()]
        top10_PB = top10_PB[top10_PB[:, 1].argsort()]

    # 绘图
    if plot_on:
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        for data1, data2 in zip(data_list_PE, data_list_PB):
            ax1.plot(data1['trade_date'], data1['pe'], label=(data1.loc[0, 'name'] + '_PE'))
            ax1.xaxis.set_major_locator(LinearLocator(numticks=8))
            plt.xticks(rotation=45)
            ax1.legend()
            ax1.set_title('PE_TOP10')

            ax2.plot(data1['trade_date'], data1['pb'], label=(data1.loc[0, 'name'] + '_PB'))
            ax2.xaxis.set_major_locator(LinearLocator(numticks=8))
            plt.xticks(rotation=45)
            ax2.legend()
            ax2.set_title('PB_TOP10')

        plt.show()


if __name__ == '__main__':

    # 单股查看
    # name = '华润双鹤'
    # ansys_sigle(name)

    # 板块查看
    industry = "化学制药"
    ansys_indus(industry=industry)















