import tushare as ts
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as dates
import pdb
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, LinearLocator
import datetime
from findN import findN
from findTREND import findTREND
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

pd.get_option('display.width')
pd.set_option('display.width', 180)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 100)

pro = ts.pro_api()


def piceGo(tscode, name, start_date, end_date):
    print(tscode, name)
    # 复权价格
    # 日线
    df = ts.pro_bar(ts_code=tscode, adj='qfq', start_date=start_date, end_date=end_date)
    df = df.sort_values(by='trade_date', ascending=True)
    df['trade_date'] = pd.to_datetime(df['trade_date'], format="%Y-%m-%d")
    plt.plot(df['trade_date'], df['close'].values, '-', label='D')

    # 周线
    df = ts.pro_bar(ts_code=tscode, freq='W', adj='qfq', start_date=start_date, end_date=end_date)
    df = df.sort_values(by='trade_date', ascending=True)
    df['trade_date'] = pd.to_datetime(df['trade_date'], format="%Y-%m-%d")

    plt.plot(df['trade_date'], df['close'].values, '-', label='W')

    plt.vlines(pd.to_datetime('20200123'), df['close'].min(), df['close'].max(), 'r')
    plt.vlines(pd.to_datetime('20200203'), df['close'].min(), df['close'].max(), 'r')


def moneyflowGO(tscode, name, start_date, end_data):
    # 个股资金流向
    # 获取单个股票数据
    df = pro.moneyflow(ts_code=tscode, start_date=start_date, end_date=end_data)
    df['trade_date'] = pd.to_datetime(df['trade_date'], format="%Y-%m-%d")
    df.set_index('trade_date', inplace=True)
    df.drop(['ts_code'], axis=1, inplace=True)
    df1 = df[['buy_sm_amount', 'sell_sm_amount', 'buy_md_amount', 'sell_md_amount', 'buy_md_amount', 'sell_md_amount', 'buy_lg_amount', 'sell_lg_amount', 'buy_elg_amount', 'sell_elg_amount']]
    df1.plot()


def holdtradeGo(tscode, name, start_date, end_data):
    df = pro.stk_holdertrade(ts_code=tscode)
    df['ann_date'] = pd.to_datetime(df['ann_date'], format="%Y-%m-%d")
    df.set_index('ann_date', inplace=True)

    print(df[['holder_name', 'change_vol', 'avg_price', 'change_ratio', 'total_share', 'after_share', 'in_de']])
    df = df[['after_share']]
    df.plot(marker='.')


if __name__ == '__main__':
    # stocks_list = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
    # stocks_list.to_csv('stocks_list.csv', index=False)
    stocks_list = pd.read_csv('stocks_list.csv')
    stocks_list = stocks_list[stocks_list['done'] != 1]

    stocks_list.set_index('name', inplace=True)
    randname = np.random.choice(stocks_list.index.values, len(stocks_list))
    tempd = pd.DataFrame({'name': randname})

    tempd = pd.merge(tempd, stocks_list, how='left', on='name')
    start_date = '2017-01-01'
    end_date = '2020-02-22'
    # 绘图
    for i, item in tempd.iterrows():
        # moneyflowGO(tscode=item['ts_code'], name=item['name'], start_date=start_date, end_data=end_date)
        # holdtradeGo(tscode=item['ts_code'], name=item['name'], start_date=start_date, end_data=end_date)

        fig = plt.figure(figsize=[9, 5])
        ax = fig.add_subplot(1, 1, 1)
        piceGo(tscode=item['ts_code'], name=item['name'], start_date=start_date, end_date=end_date)
        ax.xaxis.set_major_locator(LinearLocator(numticks=None))
        plt.legend()
        plt.grid(True)
        # plt.gcf().autofmt_xdate()
        plt.xticks(rotation=45)
        plt.title(item['ts_code'] + '_' + item['name'])
        plt.grid(True)
        # A = findN(item['name'])
        # A.plotZ()

        T = findTREND(item['name'], start_date=start_date, end_date=end_date)
        T.plotTREND()


        plt.show()