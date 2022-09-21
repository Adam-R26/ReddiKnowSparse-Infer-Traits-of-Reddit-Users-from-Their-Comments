# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 16:30:39 2022

@author: adamr
"""

import unittest
from VisualizationPlotter import VisualizationPlotter
import matplotlib

class test_visualization_plotter(unittest.TestCase):
    def test_acc_single_results(self):
        data = [{'LR':{'Accuracy': 0.7, 'F1 Score': 0.72}, 'RF':{'Accuracy': 0.7, 'F1 Score': 0.72}, 'KNN':{'Accuracy': 0.7, 'F1 Score': 0.72}, 'SVC':{'Accuracy': 0.7, 'F1 Score': 0.72}, 'CNN':{'Accuracy': 0.7, 'F1 Score': 0.72}, 'Deep CNN':{'Accuracy': 0.7, 'F1 Score': 0.72}}]
        plotter = VisualizationPlotter(data)
        graph = plotter.plot_metric_graph(['Example'], 'bar', 'example', 'algorithm', 'accuracy', 'Accuracy')
        self.assertEqual(matplotlib.figure.Figure, type(graph))
        
        
    def test_f1_single_results(self):
        data = [{'LR':{'Accuracy': 0.7, 'F1 Score': 0.72}, 'RF':{'Accuracy': 0.7, 'F1 Score': 0.72}, 'KNN':{'Accuracy': 0.7, 'F1 Score': 0.72}, 'SVC':{'Accuracy': 0.7, 'F1 Score': 0.72}, 'CNN':{'Accuracy': 0.7, 'F1 Score': 0.72}, 'Deep CNN':{'Accuracy': 0.7, 'F1 Score': 0.72}}]
        plotter = VisualizationPlotter(data)
        graph = plotter.plot_metric_graph(['Example'], 'bar', 'example', 'algorithm', 'f1 score', 'F1 Score')
        self.assertEqual(matplotlib.figure.Figure, type(graph))
    
    def test_acc_double_results(self):
        data = [{'LR':{'Accuracy': 0.7, 'F1 Score': 0.72}, 'RF':{'Accuracy': 0.7, 'F1 Score': 0.72}, 'KNN':{'Accuracy': 0.7, 'F1 Score': 0.72}, 'SVC':{'Accuracy': 0.7, 'F1 Score': 0.72}, 'CNN':{'Accuracy': 0.7, 'F1 Score': 0.72}, 'Deep CNN':{'Accuracy': 0.7, 'F1 Score': 0.72}}, {'LR':{'Accuracy': 0.7, 'F1 Score': 0.72}, 'RF':{'Accuracy': 0.7, 'F1 Score': 0.72}, 'KNN':{'Accuracy': 0.7, 'F1 Score': 0.72}, 'SVC':{'Accuracy': 0.7, 'F1 Score': 0.72}, 'CNN':{'Accuracy': 0.7, 'F1 Score': 0.72}, 'Deep CNN':{'Accuracy': 0.7, 'F1 Score': 0.72}}]
        plotter = VisualizationPlotter(data)
        graph = plotter.plot_metric_graph(['Example', 'Example 2'], 'bar', 'example', 'algorithm', 'accuracy', 'Accuracy')
        self.assertEqual(matplotlib.figure.Figure, type(graph))
        
    def test_f1_double_results(self):
        data = [{'LR':{'Accuracy': 0.7, 'F1 Score': 0.72}, 'RF':{'Accuracy': 0.7, 'F1 Score': 0.72}, 'KNN':{'Accuracy': 0.7, 'F1 Score': 0.72}, 'SVC':{'Accuracy': 0.7, 'F1 Score': 0.72}, 'CNN':{'Accuracy': 0.7, 'F1 Score': 0.72}, 'Deep CNN':{'Accuracy': 0.7, 'F1 Score': 0.72}}, {'LR':{'Accuracy': 0.7, 'F1 Score': 0.72}, 'RF':{'Accuracy': 0.7, 'F1 Score': 0.72}, 'KNN':{'Accuracy': 0.7, 'F1 Score': 0.72}, 'SVC':{'Accuracy': 0.7, 'F1 Score': 0.72}, 'CNN':{'Accuracy': 0.7, 'F1 Score': 0.72}, 'Deep CNN':{'Accuracy': 0.7, 'F1 Score': 0.72}}]
        plotter = VisualizationPlotter(data)
        graph = plotter.plot_metric_graph(['Example', 'Example 2'], 'bar', 'example', 'algorithm', 'f1 score', 'F1 Score')
        self.assertEqual(matplotlib.figure.Figure, type(graph))
        
        
if __name__ == "__main__":
    unittest.main()
