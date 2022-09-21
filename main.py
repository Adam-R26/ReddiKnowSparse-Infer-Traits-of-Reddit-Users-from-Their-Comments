# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 14:39:31 2022

@author: adamr
"""
from PipelineConfiguration import PipelineConfiguration
from ModelTrainingPipeline import ModelTrainingPipeline
import traceback
import pickle
import os

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    config_tf_uni = PipelineConfiguration(r'C:\Users\adamr\Documents\UniversityWork\COMP390 Reddit Political Sentiment Project\Hyperparam Grids\\', 
                                   r'C:\Users\adamr\Documents\UniversityWork\COMP390 Reddit Political Sentiment Project\Code\Pipeline\Data\pre_processed_data.csv',
                                   '2021-12-5', 
                                   '2021-1-1',
                                   ['tories', 'LabourUK'], 
                                   50000, 
                                   2500,
                                   True,
                                   True,
                                   further_pre_process=True
                                   )
    
    
    # config_bow_uni = PipelineConfiguration(r'C:\Users\adamr\Documents\UniversityWork\COMP390 Reddit Political Sentiment Project\Output\\', 
    #                                r'C:\Users\adamr\Documents\University Work\COMP390 Reddit Political Sentiment Project\Code\pre_processed_data.csv',
    #                                '2021-12-5', 
    #                                '2021-1-1',
    #                                ['tories', 'LabourUK'], 
    #                                50000, 
    #                                2500,
    #                                False,
    #                                False,
    #                                ngram_range=(2,2)
    #                                )
    
   
    
    
    # config_no_preprocessing_tf_uni = PipelineConfiguration(r'C:\Users\adamr\Documents\University Work\COMP390 Reddit Political Sentiment Project\Output\\', 
    #                                r'C:\Users\adamr\Documents\University Work\COMP390 Reddit Political Sentiment Project\Raw Data\data.csv',
    #                                '2021-12-5', 
    #                                '2021-1-1',
    #                                ['tories', 'LabourUK'], 
    #                                50000, 
    #                                2500,
    #                                False,
    #                                True,
    #                                further_pre_process=False, 
    #                                ngram_range=(2,2))
    
    
    
    tf_uni = ModelTrainingPipeline(config_tf_uni).main(store_preprocessed_data=False, use_api_flag=False)
    
    
    
    
    
    #bow_uni = ModelTrainingPipeline(config_bow_uni).main(store_preprocessed_data=False)
    #tf_no_pre = ModelTrainingPipeline(config_no_preprocessing_tf_uni).main(store_preprocessed_data=False)

main()