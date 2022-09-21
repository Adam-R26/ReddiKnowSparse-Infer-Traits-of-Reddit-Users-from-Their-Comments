import pandas as pd
from RedditDataAcquirer import RedditDataAcquirer
from RedditDataPreprocessor import RedditDataPreprocessor
from FeatureExtractor import FeatureExtractor
from ModelBuilder import ModelBuilder
from VisualizationPlotter import VisualizationPlotter
import pickle
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

class ModelTrainingPipeline:
    def __init__(self, config):
        self.config = config
        self.data_acquirer = RedditDataAcquirer(config.before_date, config.after_date, config.subreddit, config.number_of_comments, config.data_file_path)
        self.data_preprocessor = RedditDataPreprocessor(config.pre_process, config.class_data_points)
        self.feature_extractor = FeatureExtractor(config.further_pre_process, config.tf_idf, ngram_range=config.ngram_range)
        
    
    def main(self, use_api_flag=False, store_acquired_data=True, store_preprocessed_data=True, store_vectorizer=True, store_f1_plot=True, store_models=True, store_performance_metrics=True, store_optimal_hyperparams=True):
        #Acquire the data
        df = self.data_acquirer.acquire_data(use_api_flag)
        if store_acquired_data:
            df.to_csv(self.config.output_filepath+'raw_data.csv')
        
        
        #Pre-process the data if required
        df = self.data_preprocessor.preprocess_data(df)
        if store_preprocessed_data:
            df.to_csv(self.config.output_filepath+'pre_processed_data.csv', index=False)
        
        #Vectorize data
        X_train, X_test, Y_train, Y_test, vocab_len = self.feature_extractor.extract_bow_features(df)
        if store_vectorizer:
            pickle.dump(self.feature_extractor.vectorizer, open(self.config.output_filepath+'vectorizer.pickle', "wb"))
        
        #Train models
        model_builder = ModelBuilder(X_train, X_test, Y_train, Y_test, vocab_len)
        model_builder.train_all_models()
        if store_models:
            pickle.dump(model_builder.get_trained_models(), open(self.config.output_filepath+'models.pickle', "wb"))
    
        if store_performance_metrics:
            pickle.dump(model_builder.get_performance_metrics(), open(self.config.output_filepath+'performance_metrics.pickle', "wb"))
            
        if store_optimal_hyperparams:
            pickle.dump(model_builder.get_optimal_hyperparameters(), open(self.config.output_filepath+'optimal_hyperparamters.pickle', "wb"))
        
        
        #Build visualization
        f1_vis_plotter = VisualizationPlotter([model_builder.get_performance_metrics().copy()]) 
        acc_vis_plotter = VisualizationPlotter([model_builder.get_performance_metrics().copy()]) 
        f1_plot = f1_vis_plotter.plot_metric_graph(['F1 Score'], 'bar', 'F1 Score by Model', 'Model', 'F1 Score', 'F1 Score')
        acc_plot = acc_vis_plotter.plot_metric_graph(['Accuracy'], 'bar', 'Accuracy by Model', 'Model', 'Accuracy', 'Accuracy')
        if store_f1_plot:
            f1_plot.savefig(self.config.output_filepath+'f1_plot.png')
            acc_plot.savefig(self.config.output_filepath+'accuracy_plot.png')
        
        return model_builder
        
        
            
            
        
        
        
        
        
        
        
        
    

