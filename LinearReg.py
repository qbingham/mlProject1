from statistics import mean
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn import linear_model
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


def create_dataset():
    data_file = open("wdbc.data.txt", "r")
    xs = []
    ys = []
    for line in data_file:
        current_line = line.split(",")
        xs.append(float(current_line[2]))
        ys.append(float(current_line[4]))
    data_file.close()
    return np.array(xs, dtype=np.float64),np.array(ys,dtype=np.float64)

def best_fit_slope_and_intercept(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)*mean(xs)) - mean(xs*xs)))
    
    b = mean(ys) - m*mean(xs)

    return m, b


def coefficient_of_determination(ys_orig,ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]

    squared_error_regr = sum((ys_line - ys_orig) * (ys_line - ys_orig))
    squared_error_y_mean = sum((y_mean_line - ys_orig) * (y_mean_line - ys_orig))

    print(squared_error_regr)
    print(squared_error_y_mean)

    r_squared = 1 - (squared_error_regr/squared_error_y_mean)

    return r_squared

def use_sk():
    data = load_breast_cancer()
    xs = data.data[:, np.newaxis, 0]
    ys = data.data[:, np.newaxis, 2]
    regression = linear_model.LinearRegression()
    regression.fit(xs,ys)
    ys_pred = regression.predict(xs)
    coefficient_sklearn = r2_score(ys, ys_pred)
    print("sklearn coeff: ", coefficient_sklearn)

    plt.scatter(xs, ys, color='#000000', label='data')
    plt.plot(xs, ys_pred, label='regression line')
    # plt.xticks(())
    # plt.yticks(())
    plt.legend(loc=4)
    plt.show()

def use_np():
    xs, ys = create_dataset()
    m, b = best_fit_slope_and_intercept(xs,ys)
    regression_line = [(m*x)+b for x in xs]
    r_squared = coefficient_of_determination(ys,regression_line)
    print("numpy coeff: ", r_squared)
    plt.scatter(xs,ys,color='#003F72', label = 'data')
    plt.plot(xs, regression_line, label = 'regression line')
    plt.legend(loc=4)
    plt.show()

use_np()
use_sk()