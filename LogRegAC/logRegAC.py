import numpy as np

class LogReg:
    def __init__(self, lr=0.01, max_iter=1000, eps=1e-5, newton=False, stochGD=False):
        self.w = None
        self.cost_hist = []
        self.w_hist = []
        self.lr = lr
        self.max_iter = max_iter
        self.eps = eps
        self.newton = newton
        self.stochGD = stochGD
    
    def fit(self, X, y, iter_step=1, reg_term=None):
        if self.newton:
            print('Implementing Newton\'s Method.')
            return self.__newton(X, y, iter_step=iter_step, reg_term=reg_term)
        else:
            if self.stochGD:
                print('Implementing Stochastic Gradient Descent.')
                return self.__stochGrad(X, y, iter_step)
            else:
                print('Implementing Batch Gradient Descent.')
                return self.__batchGrad(X, y, iter_step)
            
    def __newton_step(self, X, y, w, reg_term=None):
        p = np.array(self.sigmoid(X.dot(w[:,0])), ndmin=2).T
        W = np.diag((p*(1-p))[:,0])
        hessian = X.T.dot(W).dot(X)
        grad = X.T.dot(y-p)
        if reg_term:
            step = np.dot(np.linalg.inv(hessian + reg_term*np.eye(w.shape[0])), grad)
        else:
            step = np.dot(np.linalg.inv(hessian), grad)
            
        beta = w + step
        
        return beta
            
    def __newton(self, X, y, iter_step=1, reg_term=None):
        n = X.shape[1]
        w_old, w = np.ones((n,1)), np.zeros((n,1))
        for i in range(self.max_iter + 1):
            cost = self.cost(X, y, w_old)
            self.cost_hist.append(cost)
            if i % iter_step == 0:
                print(f'Iteration {i: 5d} | Cost: {cost: 3.3f}')
            w_old = w
            w = __newton_step(X, y.to_frame(), w, reg_term)
            iter_count += 1
            if self.__check_convergence(w_old, w):
                print(f'Stopping criteria satisfied at iteration {i + 1}.')
                break
        self.w = w
    
    def __batchGrad(self, X, y, iter_step=1):
        w, X = self.init_w_x(X)
        for i in range(self.max_iter + 1):
            w_old = w.copy()
            cost = self.cost(X, y, w)
            self.cost_hist.append(cost)
            self.w_hist.append(w)
            w = w + self.lr * self.__calc_grad(X, y, w)
            if i % iter_step == 0:
                print(f'Iteration {i: 5d} | Cost: {cost: 3.3f}')
            if self.__check_convergence(w_old, w):
                print(f'Stopping criteria satisfied at iteration {i + 1}.')
                break
        self.w = w

    def __stochGrad(self, X, y, iter_step=1):
        w, X = self.init_w_x(X)
        for i in range(self.max_iter + 1):
            w_old = w.copy()
            cost = self.cost(X, y, w)
            self.cost_hist.append(cost)
            self.w_hist.append(w)
            for j in range(len(X)):
                w = w + self.lr * self.__calc_grad(X[j], y[j], w)
            if i % iter_step == 0:
                print(f'Iteration {i: 5d} | Cost: {cost: 3.3f}')
            if self.__check_convergence(w_old, w):
                print(f'Stopping criteria satisfied at iteration {i + 1}.')
                break
        self.w = w

    def __calc_grad(self, X, y, w):
        if self.stochGD:
            return np.dot(X.T, (y - self.sigmoid(np.dot(X, w))).item()).reshape(w.shape)
        return np.dot(X.T, np.reshape(y,(len(y),1)) - self.sigmoid(np.dot(X, w)))
    
    def init_w_x(self, X):
        w = np.zeros((X.shape[1] + 1, 1))
        X = np.c_[np.ones((X.shape[0], 1)), X]
        return w, X
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def cost(self, X, y, w):
        z = np.dot(X, w)
        cost1 = np.dot(y.T, np.log(self.sigmoid(z)))
        cost0 = np.dot((1 - y).T, np.log(1 - self.sigmoid(z)))
        cost = -(cost1 + cost0)
        return cost.item()

    def predict(self, X, prob=False):
        z = np.dot(self.init_w_x(X)[1], self.w)
        if prob:
            return self.sigmoid(z)
        pred = []
        for i in self.sigmoid(z):
            if i > 0.5:
                pred.append(1)
            else:
                pred.append(0)
        return pred
    
    def __check_convergence(self, w_o, w_n):
        return np.linalg.norm(w_n - w_o) < self.eps
    
    def get_cost_hist(self):
        return self.cost_hist
    
    def get_w_hist(self):
        return self.w_hist
    
    def get_params(self):
        return self.w
    
    def set_params(self, w):
        self.w = w