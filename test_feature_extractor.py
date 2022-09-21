# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 14:19:05 2022

@author: adamr
"""

import unittest
import pandas as pd
from FeatureExtractor import FeatureExtractor
import re
import numpy as np
from sklearn.utils import shuffle

class test_feature_extractor(unittest.TestCase):
    def combination_tester(self, pre_process, tf_idf):
        #Read in file shrink input size and apply feature extraction
        df = pd.read_csv(r'C:\Users\adamr\Documents\UniversityWork\COMP390 Reddit Political Sentiment Project\Raw Data\Raw Data\data.csv')
        df = shuffle(df)[0:100]
        feature_extractor = FeatureExtractor(pre_process, tf_idf)
        X_train, X_test, Y_train, Y_test, vocab_len = feature_extractor.extract_bow_features(df)
        
        #Isolate and perform check on feature names
        features = feature_extractor.vectorizer.get_feature_names()
        features = ''.join(features)
        feature_check = len(re.findall(r'[^\w\s]' ,features))
        punct_check = True if feature_check > 0 else False
        
        #Perform TF-IDF Check
        design_matrix = X_train.toarray().flatten().astype(float)
        bool_arr = np.array([val.is_integer() for val in design_matrix])
        decimal_check = False if sum(bool_arr) == len(design_matrix) else True
        
    
        return punct_check, decimal_check
          
        
    def test_tf_no_pre(self):
        punct_check, decimal_check = self.combination_tester(False, True)
        self.assertEqual(True, punct_check)
        self.assertEqual(True, decimal_check)
        
    def test_tf_pre(self):
        punct_check, decimal_check = self.combination_tester(True, True)
        self.assertEqual(False, punct_check)
        self.assertEqual(True, decimal_check)
        
        
    def test_bow_no_pre(self):
        punct_check, decimal_check = self.combination_tester(False, False)
        self.assertEqual(True, punct_check)
        self.assertEqual(False, decimal_check)
        
    def test_bow_pre(self):
        punct_check, decimal_check = self.combination_tester(True, False)
        self.assertEqual(False, punct_check)
        self.assertEqual(False, decimal_check)
    
        
if __name__ == "__main__":
    unittest.main()
