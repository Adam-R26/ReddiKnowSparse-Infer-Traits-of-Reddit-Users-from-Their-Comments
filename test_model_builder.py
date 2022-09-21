# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 16:11:59 2022

@author: adamr
"""

import unittest 
from ModelBuilder import ModelBuilder
from FeatureExtractor import FeatureExtractor
import os
from sklearn.utils import shuffle
import pandas as pd

class test_model_builder(unittest.TestCase):
    def train_models(self, tf_idf):
        #Read in file shrink input size and apply feature extraction
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        df = pd.read_csv(r'C:\Users\adamr\Documents\UniversityWork\COMP390 Reddit Political Sentiment Project\Raw Data\Raw Data\data.csv')
        df = shuffle(df)[0:100]
        feature_extractor = FeatureExtractor(False, tf_idf)
        X_train, X_test, y_train, y_test, vocab_len = feature_extractor.extract_bow_features(df)
        model_builder = ModelBuilder(X_train, X_test, y_train, y_test, vocab_len)
        model_builder.train_all_models()
        metrics = model_builder.get_performance_metrics()
        metrics_populated = len(metrics)
        metrics_check = True if metrics_populated > 0 else False
        return metrics_check
    
    
    def test_tfidf_features(self):
        check = self.train_models(True)
        self.assertEqual(True, check)
    
    def test_bow_features(self):
        check = self.train_models(False)
        self.assertEqual(True, check)

        
if __name__ == "__main__":
    unittest.main()
        
        
        