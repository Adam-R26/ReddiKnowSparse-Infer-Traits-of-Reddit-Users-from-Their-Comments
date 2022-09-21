# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 12:03:02 2022

@author: adamr
"""

import unittest
from RedditDataAcquirer import RedditDataAcquirer

class test_data_acquirer(unittest.TestCase):
    def test_one_hundred_comments(self):
        data_acquirer = RedditDataAcquirer('2021-12-5',  '2021-1-1', ['tories', 'LabourUK'], 100,  '')
        df = data_acquirer.acquire_data(True)
        self.assertEqual(200, len(df))
    
    def test_zero_comments(self):      
        data_acquirer = RedditDataAcquirer('2021-12-5',  '2021-1-1', ['tories', 'LabourUK'], 0,  '')
        try:
            df = data_acquirer.acquire_data(True)
            exception = True
        except Exception as e:
            exception = e
        
        self.assertEqual(ValueError, type(exception))
        
    def test_negative_comments(self):
        data_acquirer = RedditDataAcquirer('2021-12-5',  '2021-1-1', ['tories', 'LabourUK'], -100,  '')
        
        try:
            df = data_acquirer.acquire_data(True)
            excpetion = True
        except Exception as e:
            exception = e
        
        self.assertEqual(ValueError, type(exception))
        
    def test_valid_filepath(self):
        data_acquirer = RedditDataAcquirer('2021-12-5',  '2021-1-1', ['tories', 'LabourUK'], -100,  r'C:\Users\adamr\Documents\UniversityWork\COMP390 Reddit Political Sentiment Project\Code\Pipeline\Data\pre_processed_data.csv')
        df = data_acquirer.acquire_data(False)
        
    def test_invalid_filepath(self):
        try:
            data_acquirer = RedditDataAcquirer('2021-12-5',  '2021-1-1', ['tories', 'LabourUK'], -100,  r'ifbgui.csv')
            df = data_acquirer.acquire_data(False)
            exception = True
        except Exception as e:
            exception = e
            
        self.assertEqual(FileNotFoundError, type(exception))
        
        
if __name__ == "__main__":
    unittest.main()
        
        

        
    