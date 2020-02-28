import numpy as np
from sko.GA import GA, GA_TSP
import tushare as ts
import pandas as pd
import matplotlib.pyplot as plt
import pdb
from scipy import interpolate
from scipy.optimize import minimize
from tqdm import tqdm

class findTREND(object):
    def __init__(self, name, start_date, end_date):
        self.name = name
        self.start_date = start_date
        self.end_date = end_date

    def linear_regression(self, x, y):
        N = len(x)
        sumx = sum(x)
        sumy = sum(y)
        sumx2 = sum(x ** 2)
        sumxy = sum(x * y)
        A = np.mat([[N, sumx], [sumx, sumx2]])
        b = np.array([sumy, sumxy])
        return np.linalg.solve(A, b)

    def sigline(self, xx, yy, x0, fun, cons, c):
        res = minimize(fun, x0, method='SLSQP', constraints=cons)
        dis = (res.x[0] * xx + res.x[1] - yy) ** 2
        dis = np.vstack((dis, xx))
        dis = dis[:, dis[0, :].argsort()][:, :3]
        dis = dis[:, dis[1, :].argsort()]
        # 控制受阻点间隔的系数
        rate = (dis[1, 1:] - dis[1, :-1]).min() / (dis[1, 1:] - dis[1, :-1]).max()
        rate_the = 0.3

        if (rate > rate_the) and dis[0, :].sum() < 0.01:
            if c == 0 and res.x[0] > 0 \
                    and (self.f_inter(dis[1, 2]) > self.f_inter(dis[1, 2] + 1)) \
                    and (self.f_inter(dis[1, 2]) > self.f_inter(dis[1, 2] - 1)) \
                    and (self.f_inter(dis[1, 1]) > self.f_inter(dis[1, 1] + 1)) \
                    and (self.f_inter(dis[1, 1]) > self.f_inter(dis[1, 1] - 1)) \
                    and (self.f_inter(dis[1, 0]) > self.f_inter(dis[1, 0] + 1)) \
                    and (self.f_inter(dis[1, 0]) > self.f_inter(dis[1, 0] - 1)):
                plt.plot(dis[1, [0, -1]], dis[1, [0, -1]] * res.x[0] + res.x[1], 'g')
                # plt.show()
            elif c == 1 and res.x[0] < 0 \
                    and (self.f_inter(dis[1, 2]) < self.f_inter(dis[1, 2] + 1)) \
                    and (self.f_inter(dis[1, 2]) < self.f_inter(dis[1, 2] - 1)) \
                    and (self.f_inter(dis[1, 1]) < self.f_inter(dis[1, 1] + 1)) \
                    and (self.f_inter(dis[1, 1]) < self.f_inter(dis[1, 1] - 1)) \
                    and (self.f_inter(dis[1, 0]) < self.f_inter(dis[1, 0] + 1)) \
                    and (self.f_inter(dis[1, 0]) < self.f_inter(dis[1, 0] - 1)):
                plt.plot(dis[1, [0, -1]], dis[1, [0, -1]] * res.x[0] + res.x[1], 'r')
                # plt.show()

    def plotTREND(self):
        # 周线
        plt.figure()
        stocks_list = pd.read_csv('stocks_list.csv')
        ts_code = stocks_list[stocks_list['name'] == self.name]['ts_code'].values[0]

        df = ts.pro_bar(ts_code=ts_code, freq='d', adj='hfq', start_date=self.start_date, end_date=self.end_date)
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df[['trade_date', 'close']]
        df['trade_date'] = ((df['trade_date'] - df['trade_date'].max()) / np.timedelta64(1, 'D') / 1).astype(int)
        df = np.abs(df.values)
        data = df[df[:, 0].argsort()]
        self.f_inter = interpolate.interp1d(data[:, 0], data[:, 1], kind='slinear')
        plt.plot(df[:, 0], data[:, 1])
        for k in tqdm(range(min(100, len(data))), desc="[Findign Trend Line...]"):
            for i in range(k+10, len(data)):
                xx = data[k:i, 0]
                yy = data[k:i, 1]
                x0 = self.linear_regression(xx, yy)[::-1]
                # 包络所有高点阈值系数
                alpha = 0.01
                fun = lambda x: np.sort(((yy[1:] - x[0] * xx[1:] - x[1])**2))[:3].mean()
                cons1 = ({'type': 'ineq', 'fun': lambda x: x[0] * xx + x[1] - yy + alpha * yy.mean()})
                cons2 = ({'type': 'ineq', 'fun': lambda x: - x[0] * xx - x[1] + yy + alpha * yy.mean()})
                self.sigline(xx, yy, x0, fun, cons1, 0)
                self.sigline(xx, yy, x0, fun, cons2, 1)
        # plt.show()


if __name__ == '__main__':
    name = '新农开发'
    A = findTREND(name, '2017-01-01', '2020-02-22')
    A.plotTREND()
