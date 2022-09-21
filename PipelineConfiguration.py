# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 14:11:53 2022

@author: adamr
"""
from datetime import datetime

class PipelineConfiguration:
    def __init__(self, output_filepath, data_file_path, before_date, after_date, subreddit, number_of_comments, class_data_points, pre_process, tf_idf, ngram_range=(1,1), further_pre_process=True):
        self.output_filepath = output_filepath
        self.subreddit = subreddit
        self.number_of_comments = number_of_comments
        self.class_data_points = class_data_points
        self.before_date = before_date
        self.after_date = after_date
        self.data_file_path = data_file_path
        self.pre_process = pre_process
        self.tf_idf = tf_idf
        self.ngram_range = ngram_range
        self.further_pre_process = further_pre_process
        
        
        
        
    

    
    
        
        
    