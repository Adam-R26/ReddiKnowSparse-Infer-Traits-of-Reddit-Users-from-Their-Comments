import numpy as np

class HyperparameterGridConfigs:
    def get_rf_hyperparam_grid(self):
        max_features_range = ['sqrt', 'log2']
        n_estimators_range = np.arange(100, 600, 100)
        param_grid = dict(max_features=max_features_range, n_estimators=n_estimators_range)
        return param_grid

    def get_lr_hyperparam_grid(self):
        solvers = ['saga', 'liblinear', 'newton-cg']
        penalty = ['l2']
        c_values = [1]
        param_grid = dict(solver=solvers, penalty=penalty, C=c_values)
        return param_grid

    def get_svm_hyperparam_grid(self):
        kernel = ['rbf', 'sigmoid']
        C = [1]
        param_grid = dict(kernel=kernel, C=C)
        return param_grid

    def get_knn_hyperparam_grid(self):
        k_values = [40, 70, 100, 140]
        metrics = ['euclidean', 'minkowski']
        param_grid = dict(n_neighbors=k_values, metric=metrics)
        return param_grid
    
    