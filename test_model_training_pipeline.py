import unittest
from PipelineConfiguration import PipelineConfiguration
from ModelTrainingPipeline import ModelTrainingPipeline
import os


class test_model_training_pipeline(unittest.TestCase):
    def test_saved_objects(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        config_tf_uni = PipelineConfiguration(r'C:\Users\adamr\Documents\UniversityWork\COMP390 Reddit Political Sentiment Project\Unit Test\\', 
                                       r'C:\Users\adamr\Documents\UniversityWork\COMP390 Reddit Political Sentiment Project\Code\Pipeline\Data\pre_processed_data.csv',
                                       '2021-12-5', 
                                       '2021-1-1',
                                       ['tories', 'LabourUK'], 
                                       50000, 
                                       250,
                                       True,
                                       True,
                                       further_pre_process=True
                                       )
        tf_uni = ModelTrainingPipeline(config_tf_uni).main(use_api_flag=False)
    
    def test_no_saved_objects(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        config_tf_uni = PipelineConfiguration(r'C:\Users\adamr\Documents\UniversityWork\COMP390 Reddit Political Sentiment Project\Unit Test\\', 
                               r'C:\Users\adamr\Documents\UniversityWork\COMP390 Reddit Political Sentiment Project\Code\Pipeline\Data\pre_processed_data.csv',
                               '2021-12-5', 
                               '2021-1-1',
                               ['tories', 'LabourUK'], 
                               50000, 
                               250,
                               True,
                               True,
                               further_pre_process=True
                               )
        tf_uni = ModelTrainingPipeline(config_tf_uni).main(store_preprocessed_data=False, store_acquired_data=False, store_f1_plot=False ,use_api_flag=False, store_optimal_hyperparams=False, store_performance_metrics=False)
        
if __name__ == "__main__":
    unittest.main()


        