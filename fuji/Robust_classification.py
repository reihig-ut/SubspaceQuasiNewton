import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
import pandas as pd
import time
import os
import scipy.stats
from scipy import sparse
from sklearn import preprocessing
rc('text', usetex=True)

plt.rcParams["font.size"] = 17
plt.rcParams['axes.axisbelow'] = True
np.set_printoptions(precision=4)


class GradientDescent_solve:
    def __init__(self, fun, der, X, Y, step_size, alpha=0.5, beta=0.0, tol=1e-7, ite_max=2000, initial=0):
        self.fun = fun        # 目的関数
        self.der = der         # 関数の勾配
        self.alpha  = alpha          # Armijo条件の定数
        self.beta = beta         # 方向微係数の学習率
        self.step_size = step_size         # 方向微係数の学習率
        self.tol = tol         # 勾配ベクトルのL2ノルムがこの値より小さくなると計算を停止
        self.path = None       # 解の点列
        self.ite_max = ite_max # 最大反復回数
        self.X = X # データ
        self.Y = Y # ラベルデータ

        #self.lam = lam #lambda


    def minimize(self, w):
        results = []
        initial_time = time.time()
        params_name = params_to_dir_name(n=len(self.X[0]), l=len(self.X), alpha=self.alpha, beta=self.beta, initial=str(w[0]).replace('.', ''))
        make_result_dir('./results/'+params_name)

        for i in range(self.ite_max):
            grad = self.der(w,self.X,self.Y)
            if (i%5==0):
                results.append([i, time.time()-initial_time, self.fun(w,self.X,self.Y), np.linalg.norm(grad,2)])
                #print("正解率は " + str(compute_accuracy(self.X, self.Y, self.fun, w)*100) + "%")
                print(results[-1])
            direction = -grad
            if (np.linalg.norm(grad, ord=2)<self.tol):
                print("stop, 反復回数は" + str(i))
                break
            else:
                beta_l = 1.0
                while self.fun(w + beta_l*direction, self.X, self.Y) > (self.fun(w, self.X, self.Y) + self.alpha * beta_l*np.dot(direction, grad)):
                    # Armijo条件を満たすまでループする
                    beta_l = self.beta*beta_l

                w = w + beta_l * direction

        np.savetxt('./results/' + params_name + '/GD.txt', np.array(results), fmt = '%.11f')

        #print("反復回数は" + str(i))
        self.opt_w = w                # 最適解
        self.opt_result = self.fun(w,self.X,self.Y) # 関数の最小値
        print("正解率は " + str(compute_accuracy(self.X, self.Y, self.fun, w)*100) + "%")

        #self.path = np.array(path)    # 探索解の推移

class RNM_solve:
    def __init__(self, fun, der, hessian, X, Y, step_size, c_1=2.0, c_2=1.0, gamma=0.5, alpha=0.5, beta=0.6, tol=1e-7, ite_max=2000):
        self.fun = fun        # 目的関数
        self.der = der         # 関数の勾配
        self.hessian = hessian
        self.c_1 = c_1
        self.c_2 = c_2
        self.gamma = gamma
        self.der = der
        self.alpha  = alpha          # Armijo条件の定数
        self.beta = beta         # 方向微係数の学習率
        self.step_size = step_size         # 方向微係数の学習率
        self.tol = tol         # 勾配ベクトルのL2ノルムがこの値より小さくなると計算を停止
        self.path = None       # 解の点列
        self.ite_max = ite_max # 最大反復回数
        self.X = X # データ
        self.Y = Y # ラベルデータ
        #self.lam = lam #lambda

    def minimize(self, w):
        results = []
        initial_time = time.time()
        params_name = params_to_dir_name(n=len(self.X[0]), l=len(self.X), alpha=self.alpha, beta=self.beta, initial=str(w[0]).replace('.', ''))
        make_result_dir('./results/'+params_name)

        for i in range(self.ite_max):
            grad = self.der(w,self.X,self.Y)
            if (i%2==0):
                results.append([i, time.time()-initial_time, self.fun(w,self.X,self.Y), np.linalg.norm(grad,2)])
                print(results[-1])
            H = self.hessian(w,self.X,self.Y)
            eigens, vectors = np.linalg.eigh(H)
            min_eig_value = min(eigens)
            Lambda = max(0, -1.0*min_eig_value)
            direction = -1.0 * np.linalg.inv( (H + (self.c_1*Lambda + self.c_2*np.linalg.norm(grad,2)**self.gamma)*np.identity(len(H))) ) @ grad
            if (np.linalg.norm(grad, ord=2)<self.tol):
                print("stop, 反復回数は" + str(i))
                break
            else:
                beta_l = 1.0
                while self.fun(w + beta_l*direction, self.X, self.Y) > (self.fun(w, self.X, self.Y) + self.alpha * beta_l*np.dot(direction, grad)):
                    # Armijo条件を満たすまでループする
                    beta_l = self.beta*beta_l

                w = w + beta_l * direction

        np.savetxt('./results/' + params_name + '/RNM.txt', np.array(results), fmt = '%.11f')

        self.opt_w = w                # 最適解
        self.opt_result = self.fun(w,self.X,self.Y) # 関数の最小値
        print("正解率は " + str(compute_accuracy(self.X, self.Y, self.fun, w)*100) + "%")

        #self.path = np.array(path)    # 探索解の推移


class RS_RNM_solve:
    def __init__(self, fun, der, sketched_hessian,d, X, Y, step_size, c_1=2.0, c_2=1.0, gamma=0.5, alpha=0.5, beta=0.6, tol=1e-7, ite_max=2000):
        self.fun = fun        # 目的関数
        self.der = der         # 関数の勾配
        self.sketched_hessian = sketched_hessian
        self.d = d         # sketch size
        self.c_1 = c_1
        self.c_2 = c_2
        self.gamma = gamma
        self.der = der
        self.alpha  = alpha          # Armijo条件の定数
        self.beta = beta         # 方向微係数の学習率
        self.step_size = step_size         # 方向微係数の学習率
        self.tol = tol         # 勾配ベクトルのL2ノルムがこの値より小さくなると計算を停止
        self.path = None       # 解の点列
        self.ite_max = ite_max # 最大反復回数
        self.X = X # データ
        self.Y = Y # ラベルデータ
        #self.lam = lam #lambda

    def minimize(self, w, seed_num=0):
        np.random.seed(seed_num)
        results = []
        initial_time = time.time()
        params_name = params_to_dir_name(n=len(self.X[0]), l=len(self.X), alpha=self.alpha, beta=self.beta, initial=str(w[0]).replace('.', ''))
        make_result_dir('./results/'+params_name)

        flag=False
        for i in range(self.ite_max):
            grad = self.der(w,self.X,self.Y)
            if (i%4==0):
                results.append([i, time.time()-initial_time, self.fun(w,self.X,self.Y), np.linalg.norm(grad,2)])
                print(results[-1])
            #H = self.hessian(w,self.X,self.Y)
            P = sparse_random_matrix(self.d, len(w))
            H_bar = self.sketched_hessian(w, self.X,self.Y,P)
            eigens, vectors = np.linalg.eigh(H_bar)
            min_eig_value = min(eigens)
            Lambda = max(0, -1.0*min_eig_value)
            direction = -1.0 * P.T@(np.linalg.inv((H_bar + (self.c_1*Lambda + self.c_2*np.linalg.norm(grad,2)**self.gamma)*np.identity(self.d)))@(P@grad))
            if (np.linalg.norm(grad, ord=2)<self.tol):
                print("stop, 反復回数は" + str(i))
                break
            else:
                if (np.linalg.norm(grad,2) < 1e-4) and (not flag):
                    i_ = i
                    flag = True


                beta_l = 1.0
                if (np.linalg.norm(grad,2) < 1e-4):
                    beta_l = 1/2**(i-i_)
                while self.fun(w + beta_l*direction, self.X, self.Y) > (self.fun(w, self.X, self.Y) + self.alpha * beta_l*np.dot(direction, grad)):
                    # Armijo条件を満たすまでループする
                    beta_l = self.beta*beta_l
                w = w + beta_l * direction
                #print(beta_l)

        #print("反復回数は" + str(i))

        np.savetxt('./results/' + params_name + '/RSRNM_seed'+ str(seed_num) + '_d'+str(self.d)+'.txt', np.array(results), fmt = '%.11f')

        print("正解率は " + str(compute_accuracy(self.X, self.Y, self.fun, w)*100) + "%")
        self.opt_w = w                # 最適解
        self.opt_result = self.fun(w,self.X,self.Y) # 関数の最小値

        #self.path = np.array(path)    # 探索解の推移


def compute_accuracy(X, Y, func, w):
    ans = 0
    for i in range(len(Y)):
        Y_i_pred = discriminator(np.dot(w, X[i]))
        ans = ans + (Y_i_pred==Y[i])
    return ans/len(Y)


def sigmoid(t):
    if t>0:
        return 1.0/(1.0+np.exp(-t))
    else:
        return np.exp(t)/(1.0+np.exp(t))

def func(w,X,Y,lam=0.01):
    sum = 0.0
    for i in range(len(X)):
        sum  = sum + welsch_loss(Y[i]-np.dot(X[i],w))
    return sum / len(X) + lam * np.dot(w,w)

def f_der(w,X,Y,lam=0.01):
    sum = np.zeros(len(w))
    for i in range(len(X)):
        sum = sum - welsch_loss_der(Y[i]-np.dot(X[i],w))*X[i]
    return sum / len(X) + 2.0 * lam * w

def f_hessian(w,X,Y,lam=0.01):

    #sum = np.zeros((len(w),len(w)))
    #for i in range(len(X)):
    #    X_i = X[i]
    #    Y_i_pred = sigmoid(np.dot(w, X_i))
    #    Y_i = Y[i]
    #    sum = sum + (2.0*Y_i_pred - Y_i - 3.0*Y_i_pred*Y_i_pred + 2.0*Y_i*Y_i_pred) * Y_i_pred * (1-Y_i_pred) * np.outer(X_i, X_i)
    #print(sum/len(X))
    #return sum / len(X)

    D_ = [welsch_loss_hes(Y[m]-np.dot(X[m],w)) for m in range(len(X))]
    #print(X.T@np.diag(m)@X / len(X))
    return X.T@np.diag(D_)@X / len(X) + 2.0*lam*np.identity(len(X[0]))


def f_sketched_hessian(w,X,Y,P,lam=0.01):#return PHP^T
    D_ = [welsch_loss_hes(Y[m]-np.dot(X[m],w)) for m in range(len(X))]
    XP = X@P.T
    return XP.T@np.diag(D_)@XP / len(X) + 2.0*lam * P@P.T

def robust_loss(x, tau=-2.0, c=1.0):
    return ((2-tau)/tau) * ((((x/c)**2)/(2-tau) + 1)**(tau/2) - 1)

def robust_loss_der(x, tau=-2.0, c=1.0):
    return (x/(c**2)) * (((x/c)**2)/(2-tau) + 1)**(tau/2 - 1)

def robust_loss_hes(x, tau=-2.0, c=1.0):
    a = (1.0/(c*c)) * (((x/c)**2)/(2-tau) + 1)**(tau/2 - 1)
    return a  +  (x/(c**2)) * (((x/c)**2)/(2-tau) + 1)**(tau/2 - 2) * (tau/2.0-1) * (2.0*x/((2-tau)*(c*c)))
def robust_loss(x, tau=-2.0, c=1.0):
    return ((2-tau)/tau) * ((((x/c)**2)/(2-tau) + 1)**(tau/2) - 1)
def cauchy_loss(x):
    return np.log(x*x/2.0 + 1)
def welsch_loss(x):
    return 1-np.exp(-0.5*x*x)
def cauchy_loss_der(x):
    return 2.0*x/(x*x + 2.0)
def cauchy_loss_hes(x):
    return 2.0*(2.0-x**2)/((x*x + 2.0)**2)

def welsch_loss(x):
    return 1-np.exp(-0.5*x*x)
def welsch_loss_der(x):
    return x*np.exp(-0.5*x*x)
def welsch_loss_hes(x):
    return (1-x*x)*np.exp(-0.5*x*x)

def l1l2_loss(x):
    return np.sqrt(x*x+1) - 1.0
def l1l2_loss_der(x):
    return x / np.sqrt(x*x+1)
def l1l2_loss_hes(x):
    return 1.0 / ((x*x+1)**(3/2))


def discriminator(r):
    if (r < 0.5):
        return 0
    else:
        return 1

def sparse_random_matrix(d, n):
    p=0.05
    #X = np.zeros((d,n))
    #for i in range(d):
    #    for j in range(n):
    #        if (np.random.rand() < p):
    #            #print(np.random.rand())
    #            X[i][j] = np.random.normal(scale=1/np.sqrt(d*p))
    #return X
    return sparse.random(d, n, density=p, data_rvs=np.random.randn).A / np.sqrt(d*p)

def make_dataset():
    l_1, l_2 = 300, 300
    dataset = pd.read_csv("./ad-dataset/ad.csv", encoding="utf_8", dtype=str, header=None)
    raw_data_list = dataset.values.tolist()
    X, Y = [], []
    num_label_1=0
    num_label_2=0
    for datum in raw_data_list:
        label = (datum[-1]=='ad.')
        m = len(datum)
        err_flag = False
        for i in range(m-1):
            try:
                datum[i] = float(datum[i])
            except ValueError:
                err_flag = True
        if not err_flag:
            if(label):
                if(num_label_1<l_1):
                    X.append(datum[:1500])
                    Y.append(float(label))
                    num_label_1 += 1
            else:
                if(num_label_2<l_2):
                    X.append(datum[:1500])
                    Y.append(float(label))
                    num_label_2 += 1
    X = np.array(X)
    Y = np.array(Y)
    #mm = preprocessing.MinMaxScaler()
    #X = mm.fit_transform(X)
    #index = []
    #for j in range(len(X[0])):
    #    if (np.sum(X[:,j])!=0):
    #        index.append(j)
    #X = X[:, index]
    #X = scipy.stats.zscore(X)
    for j in range(len(X[0])):
        ave = np.average(X[:,j])
        std = np.std(X[:,j])
        if (std!=0.0):
            for i in range(len(X)):
                X[i][j] = (X[i][j]-ave) / std

    return np.array(X), np.array(Y)

def make_result_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return

def params_to_dir_name(n, l, alpha, beta, initial):
    s = 'n' + str(n) + '_l' + str(l) + '_alpha' + str(alpha).replace('.', '') + '_beta' + str(beta).replace('.', '') + '_initial' + str(initial)
    return s





if __name__ == '__main__':
    X, Y = make_dataset()
    n = len(X[0])
    X = X
    np.random.seed(0)
    #w_0 = np.ones(n)*1.5
    w_0 = np.random.rand(n)

    GD = GradientDescent_solve(fun=func, der=f_der, X=X, Y=Y, step_size=1.0,alpha=0.3,beta=0.5,tol=8e-8, ite_max=10000)#30000)
    GD.minimize(w=w_0)

    #RS_RNM = RS_RNM_solve(fun=func, der=f_der, sketched_hessian=f_sketched_hessian, d=50, X=X, Y=Y, alpha=0.3,beta=0.5, step_size=1.0, tol=1e-8, ite_max=4000)#8000)
    #RS_RNM.minimize(w=w_0,seed_num=0)

    RS_RNM = RS_RNM_solve(fun=func, der=f_der, sketched_hessian=f_sketched_hessian,d=100, X=X, Y=Y, alpha=0.3,beta=0.5, step_size=1.0, tol=8e-8, ite_max=3000)#6000)
    RS_RNM.minimize(w=w_0,seed_num=0)

    RS_RNM = RS_RNM_solve(fun=func, der=f_der, sketched_hessian=f_sketched_hessian,d=200, X=X, Y=Y, alpha=0.3,beta=0.5, step_size=1.0, tol=8e-8, ite_max=2000)#6000)
    RS_RNM.minimize(w=w_0,seed_num=0)

    RS_RNM = RS_RNM_solve(fun=func, der=f_der, sketched_hessian=f_sketched_hessian,d=400, X=X, Y=Y, alpha=0.3,beta=0.5, step_size=1.0, tol=8e-8, ite_max=1000)#6000)
    RS_RNM.minimize(w=w_0,seed_num=0)

    RNM = RNM_solve(fun=func, der=f_der, hessian=f_hessian, X=X, Y=Y, step_size=1.0,alpha=0.3,beta=0.5, tol=1e-8, ite_max=1000)#2000)
    RNM.minimize(w=w_0)
