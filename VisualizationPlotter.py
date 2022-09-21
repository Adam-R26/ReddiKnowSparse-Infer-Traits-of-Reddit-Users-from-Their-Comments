# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 14:00:25 2022

@author: adamr
"""
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, accuracy_score
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from copy import copy

class VisualizationPlotter:
    def __init__(self, metric_dicts):
        self.metric_dicts=metric_dicts
        
    
    def plot_metric_graph(self, figure_names, chart_type, title, x_label, y_label, metric):
        '''Function uses metric dict outputs of get_performance_dict to and plots them on the same graph.'''
        colour_list = ['mediumturquoise', 'grey', 'dodgerblue', 'orange', 'green']
        dfs = []
        font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}

        plt.rc('font', **font)
        #plt.rc('xtick', labelsize=20) 
        #plt.rc('ytick', labelsize=20) 
        plt.rc('axes', titlesize=30) 
        
        iterator = 0
        metric_dicts2 = copy(self.metric_dicts)
        
        for metric_dict in metric_dicts2:
            x = metric_dict.keys()
            for key in list(metric_dict.keys()):
                metric_dict[key] = metric_dict[key][metric]
            y = metric_dict.values()
            tmp_df = pd.DataFrame({'Model': x, metric:y})
            tmp_df['Feature Type'] = figure_names[iterator]
            dfs.append(tmp_df)
            iterator+=1

        data = pd.concat(dfs)
        data = data.pivot_table(index='Model', columns='Feature Type', values=metric)
        #return data
        graph = data.plot(kind=chart_type, title=title, xlabel=x_label, ylabel=y_label, 
                  ylim=(0,1), yticks=np.arange(0, 1.1, step=0.1), figsize=(16,12), 
                  color=colour_list[0:len(metric_dicts2)]).get_figure()
        plt.legend(fontsize = 12, loc=1)
        plt.show()
        return graph
            
                
    
            
            
            
        
        
        
    
    
    