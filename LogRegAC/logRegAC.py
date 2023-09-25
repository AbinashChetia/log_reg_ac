import numpy as np

class LogReg:
    def __init__(self, lr=0.01, max_iter=1000, eps=1e-5):
        self.w = None
        self.cost_hist = []
        self.w_hist = []
        self.lr = lr
        self.max_iter = max_iter
        self.eps = eps

    def init_w_x(self, X):
        w = np.zeros((X.shape[1] + 1, 1))
        X = np.c_[np.ones((X.shape[0], 1)), X]
        return w, X
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def cost(self, X, y, w):
        z = np.dot(X, w)
        cost0 = np.dot(y.T, np.log(self.sigmoid(z)))
        cost1 = np.dot((1 - y).T, np.log(1 - self.sigmoid(z)))
        cost = -(cost0 + cost1) / len(X)
        return cost.item()
    
    def fit(self, X, y, iter_step=1):
        w, X = self.init_w_x(X)
        for i in range(self.max_iter + 1):
            w_old = w.copy()
            w = w - self.lr * np.dot(X.T, self.sigmoid(np.dot(X, w)) - np.reshape(y,(len(y),1)))
            cost = self.cost(X, y, w)
            self.cost_hist.append(cost)
            self.w_hist.append(w)
            if i % iter_step == 0:
                print(f'Iteration {i: 5d} | Cost: {cost: 3.3f}')
            if self.__check_convergence(w_old, w):
                print(f'Stopping criteria satisfied at iteration {i + 1}.')
                break
        self.w = w

    def predict(self, X):
        z = np.dot(self.init_w_x(X)[1], self.w)
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