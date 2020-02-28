import pandas as pd
import tushare as ts
import pdb
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
import numpy as np
from tqdm import tqdm
import time
import os

pd.get_option('display.width')
pd.set_option('display.width', 180)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 100)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
pro = ts.pro_api()


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
            orig_data['ts_code'] = ts_code

            # 总股本
            total_share = pro.daily_basic(ts_code=ts_code, fields='trade_date,total_share,pe, pb')
            orig_data = pd.merge(orig_data, total_share, how='outer', on='trade_date')

            # 商誉
            goodwill = pro.balancesheet(ts_code=ts_code, fields='ann_date, goodwill')
            goodwill.rename(columns={'ann_date': 'trade_date'}, inplace=True)
            goodwill = goodwill.drop_duplicates()
            orig_data = pd.merge(orig_data, goodwill, how='outer', on='trade_date')

            # 基本每股收益, 每股净资产
            ps = pro.fina_indicator(ts_code=ts_code, fields='ann_date, eps, bps')
            ps.rename(columns={'ann_date': 'trade_date'}, inplace=True)
            ps = ps.drop_duplicates()
            orig_data = pd.merge(orig_data, ps, how='outer', on='trade_date')

            break
        except:
            print('读取频率太高，10秒后重新进行读取')
            time.sleep(10)

    # # 当前股价/基本每股收益
    # orig_data['指标1'] = orig_data['close'] / orig_data['eps']
    #
    # # 当前股价/每股净资产
    # orig_data['指标2'] = orig_data['close'] / orig_data['bps']
    # 按时间排序
    orig_data = orig_data.sort_values(by="trade_date", ascending=True)
    # 填充NaN
    orig_data = orig_data.fillna(method='ffill').fillna(method='bfill')
    # 转化成时间格式
    orig_data['trade_date'] = pd.to_datetime(orig_data['trade_date'], format="%Y-%m-%d")
    # 绘图
    if plot_on:
        # 绘图
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
        plt.show()

    return industry, orig_data


def ansys_indus(industry, plot_on=True):
    stocks_list = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
    stocks_list = stocks_list[stocks_list['industry'] == industry]
    top10_PE = np.ones((10, 2)) * 1e99
    top10_PB = np.ones((10, 2)) * 1e99
    data_list_PE = [pd.DataFrame()] * 10
    data_list_PB = [pd.DataFrame()] * 10

    print('%s 板块共计 %d 只股票，正在排序中...' % (industry, len(stocks_list)))

    for i, val in tqdm(stocks_list.iterrows()):
        _, data = ansys_sigle(name=val['name'], ts_code=val['ts_code'], industry=val['industry'], plot_on=False)
        pe, pb = data.iloc[-1, :][['pe', 'pb']].values
        # 筛选
        if (pe is not None) and (pe < top10_PE[-1, 0]):
            top10_PE[-1, :] = [pe, pb]
            data_list_PE[-1] = data

        if (pb is not None) and (pb < top10_PB[-1, 1]):
            top10_PB[-1, :] = [pe, pb]
            data_list_PB[-1] = data
        # 排序
        data_list_PE = [data_list_PE[x] for x in top10_PE[:, 0].argsort()]
        data_list_PB = [data_list_PB[x] for x in top10_PB[:, 1].argsort()]
        top10_PE = top10_PE[top10_PE[:, 0].argsort()]
        top10_PB = top10_PB[top10_PB[:, 1].argsort()]

    # 表格输出
    top10_PE_DF = pd.DataFrame(columns=['PE_rank', 'name', 'ts_code', 'PE'])
    top10_PB_DF = pd.DataFrame(columns=['PB_rank', 'name', 'ts_code', 'PB'])
    rank = 1
    for data1, data2 in zip(data_list_PE, data_list_PB):
        top10_PE_DF = top10_PE_DF.append(pd.DataFrame({'PE_rank': [rank],
                                                       'name': [data1.loc[0, 'name']],
                                                       'ts_code': [data1.loc[0, 'ts_code']],
                                                       'PE': [data1.loc[-1, 'pe']],
                                                       'PB': [data1.loc[-1, 'pb']]
                                                       }))
        top10_PB_DF = top10_PB_DF.append(pd.DataFrame({'PB_rank': [rank],
                                                       'name': [data2.loc[0, 'name']],
                                                       'ts_code': [data2.loc[0, 'ts_code']],
                                                       'PE': [data2.loc[-1, 'pe']],
                                                       'PB': [data2.loc[-1, 'pb']]
                                                         }))
    print(top10_PE_DF)
    print(top10_PB_DF)

    # 保存文件
    path = os.path.join(os.getcwd(), 'basis_ansys_result\\top10_PE.csv')
    top10_PE_DF.to_csv(path, index=False)
    path = os.path.join(os.getcwd(), 'basis_ansys_result\\top10_PB.csv')
    top10_PB_DF.to_csv(path, index=False)

    # 绘图
    if plot_on:
        fig = plt.figure(industry)
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        rank = 1
        for data1, data2 in zip(data_list_PE, data_list_PB):
            ax1.plot(data1['trade_date'], data1['pe'], label=(str(rank)) + '_' + data1.loc[0, 'name'])
            ax2.plot(data2['trade_date'], data2['pb'], label=(str(rank)) + '_' + data2.loc[0, 'name'])
            rank += 1

        ax1.legend()
        ax1.set_title('PE_TOP10')
        ax1.xaxis.set_major_locator(LinearLocator(numticks=8))
        for tick in ax1.get_xticklabels():
            tick.set_rotation(45)
        ax2.legend()
        ax2.set_title('PB_TOP10')
        ax2.xaxis.set_major_locator(LinearLocator(numticks=8))
        for tick in ax2.get_xticklabels():
            tick.set_rotation(45)
        plt.show()


if __name__ == '__main__':
    # 单股查看
    name = '通化金马'
    ansys_sigle(name)

    # 板块查看
    industry = "化学制药"
    ansys_indus(industry=industry)
