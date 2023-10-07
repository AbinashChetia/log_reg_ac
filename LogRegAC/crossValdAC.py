import numpy as np
import pandas as pd
import LogRegAC.logRegAC as logRegAC
import LogRegAC.utilsAC as utilsAC

class CrossVald:
    def __init__(self, option='holdout', monte_carlo=0, k_fold=0):
        self.option = option
        self.monte_carlo = monte_carlo
        self.k_fold = k_fold

    def fit(self, X, y, lr, max_iter, iter_step, eps, stochGD=False):
        if self.option == 'holdout':
            return self.__holdoutAC(X, y, lr, max_iter, iter_step, eps, stochGD)
        elif self.option == 'monte_carlo':
            return self.__monte_carlo(X, y, lr, max_iter, iter_step, eps, stochGD)
        elif self.option == 'k_fold':
            return self.__k_fold(X, y, lr, max_iter, iter_step, eps, stochGD)
        else:
            raise ValueError('Unknown option!')

    def __holdoutAC(self, X, y, lr, max_iter, iter_step, eps, stochGD=False, w_hist=False):
        if self.option != 'holdout':
            raise ValueError('Unknown option!')
        print('Implementing Holdout Cross Validation.')
        train_costs = []
        opt_model = {'f1': 1e-10, 'lr': None, 'w': None}
        for l in lr:
            train_x, train_y, test_x, test_y = utilsAC.splitTrainTest(X, y, 0.7)
            train_x, train_min, train_max = utilsAC.normMinMax(train_x, mode='train')
            test_x = utilsAC.normMinMax(test_x, mode='test', train_min=train_min, train_max=train_max)
            print(f'----------------- lr : {l} -----------------')
            logReg = logRegAC.LogReg(lr=l, max_iter=max_iter, eps=eps, stochGD=stochGD)
            logReg.fit(train_x, train_y, iter_step=iter_step)
            pred = logReg.predict(test_x)
            f1_temp = utilsAC.get_performance_measure(test_y, pred)['f1']
            print(f'F1-score: {f1_temp}')
            train_costs.append(logReg.get_cost_hist())
            if f1_temp > opt_model['f1']:
                opt_model['f1'] = f1_temp
                opt_model['lr'] = l
                opt_model['w'] = logReg.get_params()
        return train_costs, opt_model
    
    def __monte_carlo(self, X, y, lr, max_iter, iter_step, eps, stochGD=False):
        if self.option != 'monte_carlo':
            raise ValueError('Unknown option!')
        if self.monte_carlo == 0:
            raise ValueError('Number of iterations for Monte Carlo Cross Validation not specified!')
        print(f'Implementing Monte Carlo Cross Validation with n = {self.monte_carlo}.')
        train_costs = []
        opt_model = {'f1': 1e-10, 'lr': None, 'w': None}
        for l in lr:
            montc_train_costs = []
            montc_f1 = []
            for _ in range(self.monte_carlo):
                train_x, train_y, test_x, test_y = utilsAC.splitTrainTest(X, y, 0.7)
                train_x, train_min, train_max = utilsAC.normMinMax(train_x, mode='train')
                test_x = utilsAC.normMinMax(test_x, mode='test', train_min=train_min, train_max=train_max)
                print(f'----------------- lr : {l} -----------------')
                logReg = logRegAC.LogReg(lr=l, max_iter=max_iter, eps=eps, stochGD=stochGD)
                logReg.fit(train_x, train_y, iter_step=iter_step)
                pred = logReg.predict(test_x)
                montc_f1.append(utilsAC.get_performance_measure(test_y, pred)['f1'])
                print(f"F1-score: {utilsAC.get_performance_measure(test_y, pred)['f1']}")
                montc_train_costs.append(logReg.get_cost_hist())
            train_costs.append(montc_train_costs)
            f1_temp = np.mean(montc_f1)
            print(f'Average F1-score for lr={l}: {f1_temp}')
            if f1_temp > opt_model['f1']:
                opt_model['f1'] = f1_temp
                opt_model['lr'] = l
                opt_model['w'] = logReg.get_params()
        return train_costs, opt_model
    
    def __k_fold(self, X, y, lr, max_iter, iter_step, eps, stochGD=False):
        if self.option != 'k_fold':
            raise ValueError('Unknown option!')
        if self.k_fold == 0:
            raise ValueError('Number of folds for K-Fold Cross Validation not specified!')
        print(f'Implementing K-Fold Cross Validation with k = {self.k_fold}.')
        train_costs = []
        opt_model = {'f1': 1e-10, 'lr': None, 'w': None}
        for l in lr:
            kfold_train_costs = []
            kfold_f1 = []
            data_folds = utilsAC.split_kfold(X, y, self.k_fold)
            for i in range(self.k_fold):
                train_x = pd.DataFrame()
                train_y = pd.Series()
                for j in range(self.k_fold):
                    if j != i:
                        train_x = pd.concat([train_x, data_folds[j].iloc[:, :-1]], axis=0)
                        train_y = pd.concat([train_y, data_folds[j].iloc[:, -1]], axis=0)
                test_x = data_folds[i].iloc[:, :-1]
                test_y = data_folds[i].iloc[:, -1]
                train_x, train_min, train_max = utilsAC.normMinMax(train_x, mode='train')
                test_x = utilsAC.normMinMax(test_x, mode='test', train_min=train_min, train_max=train_max)
                train_x, train_y, test_x, test_y = train_x.reset_index(drop=True), train_y.reset_index(drop=True), test_x.reset_index(drop=True), test_y.reset_index(drop=True)
                print(f'----------------- lr : {l} -----------------')
                logReg = logRegAC.LogReg(lr=l, max_iter=max_iter, eps=eps, stochGD=stochGD)
                logReg.fit(train_x, train_y, iter_step=iter_step)
                pred = logReg.predict(test_x)
                kfold_f1.append(utilsAC.get_performance_measure(test_y, pred)['f1'])
                print(f"F1-score: {utilsAC.get_performance_measure(test_y, pred)['f1']}")
                kfold_train_costs.append(logReg.get_cost_hist())
            train_costs.append(kfold_train_costs)
            f1_temp = np.mean(kfold_f1)
            print(f'Average F1-score for lr={l}: {f1_temp}')
            if f1_temp > opt_model['f1']:
                opt_model['f1'] = f1_temp
                opt_model['lr'] = l
                opt_model['w'] = logReg.get_params()
        return train_costs, opt_model