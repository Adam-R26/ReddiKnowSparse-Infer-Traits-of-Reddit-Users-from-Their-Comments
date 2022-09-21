import tensorflow as tf
import numpy as np
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input
from tensorflow.keras.models import Sequential
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import ClassifierMixin
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score
from HyperparameterGridConfigs import HyperparameterGridConfigs


#Reproducible results for NN
tf.random.set_seed(123)

class ModelBuilder:
    def __init__(self, X_train, X_test, y_train, y_test, vocab_length):
        self._hyperparameterConfig = HyperparameterGridConfigs()
        self._X_train = X_train
        self._X_test = X_test
        self._y_train = y_train
        self._y_test = y_test
        self._vocab_length = vocab_length
        self._optimal_hyperparamters = {}
        self._trained_models = {}
        self._performance_metrics = {}
        
        
    def get_optimal_hyperparameters(self) -> dict:
        if self._optimal_hyperparamters == None:
            print('Optimal Hyperparamters not found yet, please train models to get these.')
        else:
            return self._optimal_hyperparamters
    
    def get_trained_models(self) -> dict:
        if self._trained_models == None:
            print('Models not yet train, please run train models function to get these.')
        else:
            return self._trained_models
        
    def get_performance_metrics(self) -> dict:
        if self._performance_metrics == None:
            print('Performance metrics not set yet please train models to get these.')
        else:
            return self._performance_metrics
        
    
    def _configure_cnn_model(self, vocab_len:int) -> Sequential:
        '''Builds CNN model using keras layers.'''
        model = Sequential()
        model.add(Input(batch_shape=(None, vocab_len, 1)))
        model.add(Dropout(0.3))
        model.add(Conv1D(filters=16, kernel_size=3, padding='valid', activation='relu'))
        model.add(MaxPooling1D())
        model.add(Flatten())
        model.add(Dense(250, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.build()
        model.summary()
        return model


    def _configure_deep_cnn_model(self, vocab_len:int) -> Sequential:
        '''Builds Deep CNN model using keras layers'''
        model = Sequential()
        model.add(Input(batch_shape=(None, vocab_len, 1)))
        model.add(Dropout(0.3))
        model.add(Conv1D(filters=16, kernel_size=3, padding='valid', activation='relu'))
        model.add(Conv1D(filters=16, kernel_size=3, padding='valid', activation='relu'))
        model.add(MaxPooling1D())
        model.add(MaxPooling1D())
        model.add(Flatten())
        model.add(Dense(250, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.build()
        model.summary()
        return model
    
    
    def _tune_model_hyperparamters(self) -> None:
        '''Pipelines to train all machine learning models and output the results, returns the models in a dictionary.'''
        print("----------------------------------------------------------------------")
        print("Optimizing Hyperparamters for Each Model")
        print('Random Forest:')
        rf = self._hyperparamter_tuning(RandomForestClassifier(random_state=123, n_jobs=7), self._hyperparameterConfig.get_rf_hyperparam_grid(), self._X_train, self._y_train)
        
        print('\nLogistic Regression:')
        lr = self._hyperparamter_tuning(LogisticRegression(random_state=123, n_jobs = 7, verbose=True), self._hyperparameterConfig.get_lr_hyperparam_grid(), self._X_train, self._y_train)
        
        print('\nSupport Vector Machine:')
        svm = self._hyperparamter_tuning(SVC(random_state=123, verbose=True), self._hyperparameterConfig.get_svm_hyperparam_grid(), self._X_train, self._y_train)
        
        print('\nK-Nearest Neighbour:')
        knn = self._hyperparamter_tuning(KNeighborsClassifier(), self._hyperparameterConfig.get_knn_hyperparam_grid(), self._X_train, self._y_train)
        
        print("----------------------------------------------------------------------")
        return {'RF': rf, 'LR': lr, 'SVM': svm, 'KNN': knn}
    
    
    def train_all_models(self) -> None:
        '''Trains all models using their optimal hyperparameters found using 3 repeat 5-fold cross validation'''
        self._best_hyperparamters = self._tune_model_hyperparamters()
        best_hyperparamters = self._best_hyperparamters
        
        print('BEST PARAMTERS: ', best_hyperparamters)
        rf = RandomForestClassifier(random_state=123, n_jobs=7, n_estimators=best_hyperparamters['RF']['n_estimators'], max_features=best_hyperparamters['RF']['max_features'])
        rf.fit(self._X_train, self._y_train)
        
        lr = LogisticRegression(random_state=123, C=best_hyperparamters['LR']['C'], penalty=best_hyperparamters['LR']['penalty'], solver=best_hyperparamters['LR']['solver'])
        lr.fit(self._X_train, self._y_train)
        
        knn = KNeighborsClassifier(n_jobs=7, metric=best_hyperparamters['KNN']['metric'], n_neighbors=best_hyperparamters['KNN']['n_neighbors'])
        knn.fit(self._X_train, self._y_train)
        
        svc = SVC(random_state=123, C=best_hyperparamters['SVM']['C'], kernel=best_hyperparamters['SVM']['kernel'], probability=True)
        svc.fit(self._X_train, self._y_train)
        
        dl_models = self._train_dl_models()
        
        models = {'RF': rf, 'LR': lr, 'KNN': knn, 'SVC':svc, 'CNN': dl_models['CNN'], 'Deep CNN': dl_models['Deep CNN']}
        self._trained_models = models
        
        performance_metrics = self._compute_performance_dict({'RF': rf, 'LR': lr, 'KNN': knn, 'SVC':svc})
        self._performance_metrics = dict(self._performance_metrics, **performance_metrics)
        

    def _hyperparamter_tuning(self, model: ClassifierMixin, hyperparameter_grid: dict, training_data: np.array, training_labels: np.array) -> dict:
        '''Optimizes hyperparamters for inputted model using repeated stratified k-folds cross validation, returning these optimal hyperparameters'''
        #Set up grid search with 5 fold cross validation.
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
        grid_search = GridSearchCV(estimator=model, param_grid=hyperparameter_grid, n_jobs=7, cv=cv, scoring='f1', error_score=0)
        
        #Execute grid search
        grid_result = grid_search.fit(training_data, training_labels)
        
        #Summarize the results
        print("Model Training Performance:")
        print('------------------------------------------------------------------')
        print(f"F1: {grid_result.best_score_:3f} using {grid_result.best_params_}")
        print('------------------------------------------------------------------')
        
        parameters = grid_result.best_params_
        
        return parameters
    
    def _model_testing_framework(self, trained_model: ClassifierMixin, X_test: np.array, y_test: np.array):
        '''Computes performance metrics for sklearn based models'''
        y_pred = trained_model.predict(X_test)
        y_true = y_test
        
        f1 = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        
        performance = {'F1 Score': f1,
                      'Accuracy': accuracy,
                      'Precision': precision,
                      'Recall': recall,
                      'AUC': auc}
        return performance
    
    def _compute_performance_dict(self, model_dict: dict) -> dict:
        '''Given a dictionary of trained sklearn models computes the performance metrics for each one and returns these. '''
        for model in list(model_dict.keys()):
            model_dict[model] = self._model_testing_framework(model_dict[model], self._X_test, self._y_test)
        return model_dict
    
    def _train_dl_models(self) -> dict:
        '''Uses keras from tensorflow to invoke, train and benchmark the CNN and Deep CNN'''
        #Re-shape predictor samples
        X_train = self._X_train.toarray().reshape(self._X_train.shape[0], self._X_train.shape[1], 1)
        X_test = self._X_test.toarray().reshape(self._X_test.shape[0], self._X_test.shape[1], 1)
        earlystopping = callbacks.EarlyStopping(monitor ="val_loss", 
                                        mode ="min", patience = 4, 
                                        restore_best_weights = True)
            
        #Train models 
        print('Training CNN')
        cnn = self._configure_cnn_model(self._vocab_length)
        cnn.fit(X_train, self._y_train , epochs=10, batch_size=16, verbose=1, validation_split=0.12, callbacks =[earlystopping]) #validation_data=(X_test, self._y_test)
        
        print('\n\n\nTraining Deep CNN')
        deep_cnn = self._configure_deep_cnn_model(self._vocab_length)
        deep_cnn.fit(X_train, self._y_train, epochs=10, batch_size=16, verbose=1, validation_split=0.12, callbacks =[earlystopping])

        models = {'CNN': cnn,'Deep CNN': deep_cnn}
        performance_metrics = self._compute_performance_dict_dl(models, X_test, self._y_test)
        self._performance_metrics = performance_metrics
        
        return models
    
    def _model_testing_framework_dl(self, trained_model: ClassifierMixin(), X_test: np.array, y_test: np.array) -> dict:
       '''Computes performance metrics for keras based deep learning models.'''
       y_pred = trained_model.predict(X_test)
       y_pred = [round(i[0]) for i in y_pred]
       y_true = y_test
       
       f1 = f1_score(y_true, y_pred)
       accuracy = accuracy_score(y_true, y_pred)
       precision = precision_score(y_true, y_pred)
       recall = recall_score(y_true, y_pred)
       auc = roc_auc_score(y_true, y_pred)
       
       performance = {'F1 Score': f1,
                     'Accuracy': accuracy,
                     'Precision': precision,
                     'Recall': recall,
                     'AUC': auc}
       return performance
    
    
    def _compute_performance_dict_dl(self, model_dict: dict, X_test: np.array, y_test: np.array) -> dict:
        '''Given a dictionary of trained keras models computes the performance metrics for each one and returns these. '''
        for model in list(model_dict.keys()):
            model_dict[model] = self._model_testing_framework_dl(model_dict[model], X_test, y_test)
        
        return model_dict