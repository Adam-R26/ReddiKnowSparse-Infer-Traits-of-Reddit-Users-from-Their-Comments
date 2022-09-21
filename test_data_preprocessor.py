# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 13:40:53 2022

@author: adamr
"""
import pandas as pd
import unittest
from RedditDataPreprocessor import RedditDataPreprocessor

class test_data_preprocessor(unittest.TestCase):
    def test_one_hundred_comments(self):
        '''Note this test case also covers pre_process=True'''
        df = pd.read_csv(r'C:\Users\adamr\Documents\UniversityWork\COMP390 Reddit Political Sentiment Project\Raw Data\Raw Data\data.csv')
        data_preprocessor = RedditDataPreprocessor(True, 100)
        df = data_preprocessor.preprocess_data(df)
        
        df_check = df['body'].str.contains(r'[^\w\s]', regex=True).sum()
        
        self.assertEqual(len(df), 200)
        self.assertEqual(df_check, 0)
    
    def test_negative_hundred_comments(self):
        df = pd.read_csv(r'C:\Users\adamr\Documents\UniversityWork\COMP390 Reddit Political Sentiment Project\Raw Data\Raw Data\data.csv')
        data_preprocessor = RedditDataPreprocessor(True, -100)
        
        try:
            df = data_preprocessor.preprocess_data(df)
            exception = True
        except Exception as e:
            exception = e
        
        self.assertEqual(ValueError, type(exception))
            
        
    def test_zero_comments(self):
        df = pd.read_csv(r'C:\Users\adamr\Documents\UniversityWork\COMP390 Reddit Political Sentiment Project\Raw Data\Raw Data\data.csv')
        data_preprocessor = RedditDataPreprocessor(True, 0)
        
        try:
            df = data_preprocessor.preprocess_data(df)
            exception = True
        except Exception as e:
            exception = e
        
        self.assertEqual(ValueError, type(exception))
        
    def test_no_preprocess(self):
        df = pd.read_csv(r'C:\Users\adamr\Documents\UniversityWork\COMP390 Reddit Political Sentiment Project\Raw Data\Raw Data\data.csv')
        data_preprocessor = RedditDataPreprocessor(False, 100)
        df = data_preprocessor.preprocess_data(df)
        
        df_check = df['body'].str.contains(r'[^\w\s]', regex=True).sum()
        df_check = True if df_check >=0 else False
        
        self.assertEqual(len(df), 200)
        self.assertEqual(df_check, True)
        
    
        
if __name__ == "__main__":
    unittest.main()