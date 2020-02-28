import numpy as np
from sko.GA import GA, GA_TSP
import tushare as ts
import pandas as pd
import matplotlib.pyplot as plt
import pdb
from scipy import interpolate

demo_func = lambda x: x[0] ** 2 + (x[1] - 0.05) ** 2 + (x[2] - 0.5) ** 2


class myGA(GA):
    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        for i in range(self.max_iter):
            self.X = self.chrom2x(self.Chrom)
            self.Y = self.x2y()
            self.ranking()
            self.selection()
            self.crossover()
            self.mutation()

            # record the best ones
            generation_best_index = self.FitV.argmax()
            self.generation_best_X.append(self.X[generation_best_index, :])
            self.generation_best_Y.append(self.Y[generation_best_index])
            self.all_history_Y.append(self.Y)
            self.all_history_FitV.append(self.FitV)
            # print(self.Y.min())
        global_best_index = np.array(self.generation_best_Y).argmin()
        global_best_X = self.generation_best_X[global_best_index]
        global_best_Y = self.func(np.array([global_best_X]))
        return global_best_X, global_best_Y



class findN(object):
    def __init__(self, name):
        self.name = name

    def linear_regression(self, x, y):
        N = len(x)
        sumx = sum(x)
        sumy = sum(y)
        sumx2 = sum(x ** 2)
        sumxy = sum(x * y)
        A = np.mat([[N, sumx], [sumx, sumx2]])
        b = np.array([sumy, sumxy])
        return np.linalg.solve(A, b)

    def errror(self, delta1, delta2, delta3, d1, d2):
        # 折点坐标
        subNloc = np.array([[x1, y1],
                            [x1 + delta1, y1 - d1],
                            [x1 + delta1 + delta2, y1],
                            [x1 + delta1 + delta2 + delta3, y1 - d2],
                            ])
        dis = 0
        # print(loc)
        rangmax = 0
        for i in range(3):
            # 区间坐标
            xx = np.arange(subNloc[i, 0], subNloc[i + 1, 0], 0.1)
            yy = f_inter(xx)
            if yy.max() > rangmax:
                rangmax = yy.max()
            k = (subNloc[i + 1, 1] - subNloc[i, 1]) / (subNloc[i + 1, 0] - subNloc[i, 0])
            b = subNloc[i + 1, 1] - k * subNloc[i + 1, 0]
            dis += ((yy - k * xx - b) ** 2).mean()

        #     ## 可视化
        #     plt.plot(xx, k * xx + b)
        #     plt.plot(xx, yy, 'p')
        # print(dis)
        # plt.plot(data[:, 0], data[:, 1])
        # plt.show()
        # ##

        return dis + rangmax - y1

    def meqfun(self, index, delta):
        subdata = self.data[index + 1:index + delta, :]
        index_max = np.argmax(subdata[:, 1])
        index_min = np.argmin(subdata[:, 1])

        locxy = np.array([[self.data[index, 0], self.data[index, 1]],
                          [subdata[index_max, 0], subdata[index_max, 1]],
                          [subdata[index_min, 0], subdata[index_min, 1]],
                          [self.data[index + delta, 0], self.data[index + delta, 1]]])

        locxy = locxy[locxy[:, 0].argsort()]

        if (locxy[0, 1] <= locxy[1, 1]) and (locxy[0, 1] <= locxy[2, 1]) and (locxy[2, 1] <= locxy[1, 1]) and (
                locxy[3, 1] >= locxy[1, 1]):
            # plt.plot(self.data[:, 0], self.data[:, 1], linewidth=0.5, color='lightgray')
            plt.plot(locxy[:, 0], locxy[:, 1], '-', color='g', linewidth=0.1)
            # plt.show()
            return 0, locxy

        elif (locxy[0, 1] >= locxy[1, 1]) and (locxy[0, 1] >= locxy[2, 1]) and (locxy[2, 1] >= locxy[1, 1]) and (
                locxy[3, 1] <= locxy[1, 1]) and ((locxy[0, 1] - locxy[2, 1]) < 0.1 * locxy[0, 1]):
            # plt.plot(self.data[:, 0], self.data[:, 1], linewidth=0.5, color='lightgray')
            plt.plot(locxy[:, 0], locxy[:, 1], '-', color='r', linewidth=0.1)
            # plt.show()
            # print(1)
            return 1, locxy
        return -1, []

    def returntest(self):
        plt.figure()
        for thead in range(95, 100, 1):
            # 初始钱
            money = 100000
            # 每次买股花费
            moneyGO = 1000
            # 初始股票数
            val = 0
            # 最多钱数
            money_min = 999
            # 实时价值
            value = np.zeros((len(self.tradelist), 2))
            hist_pice = 0
            for i, trade in enumerate(self.tradelist):
                if (money > 0) and trade[0] == 1:  # 买
                    # 增加股票数
                    val += moneyGO / trade[1]
                    # 减少钱数
                    money -= moneyGO
                    if money < money_min:
                        money_min = money
                elif (val > 0) and (trade[0] == -1):  # 卖
                    # 增加钱数
                    money += val * trade[1]
                    # 减少股票数
                    val = 0

                value[i, :] = [trade[2], money + val * trade[1]]

                if trade[1] < (hist_pice * thead / 100):
                    # 止损
                    # 增加钱数
                    money += val * trade[1]
                    # 减少股票数
                    val = 0
                hist_pice = trade[1]
            plt.plot(value[:, 0], value[:, 1], label=str(thead))
            plt.show()

    def plotZ(self):
        # 周线
        start_date = '2017-01-01'
        end_date = '2020-02-22'
        pro = ts.pro_api()
        stocks_list = pd.read_csv('stocks_list.csv')
        ts_code = stocks_list[stocks_list['name'] == self.name]['ts_code'].values[0]

        df = ts.pro_bar(ts_code=ts_code, freq='D', adj='hfq', start_date=start_date, end_date=end_date)
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df[['trade_date', 'close']]
        df['trade_date'] = ((df['trade_date'] - df['trade_date'].max()) / np.timedelta64(1, 'D') / 1).astype(int)
        df = np.abs(df.values)
        f_inter = interpolate.interp1d(df[:, 0], df[:, 1], kind='slinear')
        self.data = df[df[:, 0].argsort()]
        # f_inter = interpolate.interp1d(self.data[:, 0], self.data[:, 1], kind='slinear')
        self.data_index = np.arange(len(self.data))[:-30]
        # np.random.shuffle(self.data_index)
        plt.plot(self.data[:, 0], self.data[:, 1], linewidth=0.1, color='lightgray')

        tradelist = np.zeros((len(self.data_index), 3))
        # 原self.data数据的索引点（相对时间）
        for index in self.data_index:
            x1, y1 = self.data[index, :]
            Nloc = []
            u_d = []
            for j in range(4, 30):
                u_d, Nloc = self.meqfun(index, j)

                if u_d == 1:
                    # plt.plot(self.data[:, 0].max() - Nloc[:, 0], Nloc[:, 1], '-', linewidth=0.5, color='r')
                    plt.plot(Nloc[0, 0], Nloc[0, 1], '.', color='r', MarkerSize=0.1)
                    tradelist[index, :] = [1, self.data[index, 1], self.data[index, 0]]
                    break
                elif u_d == 0:
                    # plt.plot(self.data[:, 0].max() - Nloc[:, 0], Nloc[:, 1], '-', linewidth=0.5, color='g')
                    plt.plot(Nloc[0, 0], Nloc[0, 1], '.', color='g', MarkerSize=0.1)
                    tradelist[index, :] = [-1, self.data[index, 1], self.data[index, 0]]
                    break
                else:
                    tradelist[index, :] = [0, self.data[index, 1], self.data[index, 0]]
        self.tradelist = tradelist[::-1]
        # 保存图片
        filename = './result/fig_' + str(self.name) + '.eps'
        plt.savefig(filename, dpi=1200, format='eps')


        # plt.text(self.data[:, 0].max() - Nloc[0, 0] + 3, Nloc[0, 1], '%.1f' % best_y[0], ha='center', va='bottom', fontsize=3)


if __name__ == '__main__':
    name = '华润双鹤'
    A = findN(name)
    A.plotZ()
