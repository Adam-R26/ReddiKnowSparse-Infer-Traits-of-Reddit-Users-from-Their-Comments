# ReddiKnowSparse: Infer Traits of Reddit Users from their Comments
### Introduction
- This project develops an end-to-end machine learning pipeline to deduce political sentiment from Reddit comments, used as an introductory piece to Natural Language Processing (NLP), showing the implementation of traditonal vectorization methods, whilst their highlighting the importance of pre-processing and augmentation techniques such as TF-IDF in this setting.
- With social media's growing influence on political discourse, this projects capability to analyze sentiments can benefit political campaign optimization, social media post-regulation, and understanding factors affecting public political sentiment.
- It can also be used to perform binary classification between any two subreddits of choice.
- The results of this project demonstrate the performance increases we can get from pre-processing and term frequency inverse document frequency (TF-IDF) in the traditional NLP setting utilizing bag of words based feature vectors. It also shows the curse of dimensionality in effect for the K-Nearest Neighbours algorithm, showing poor performance in this domain.

### Technical Overview
### Data Acquisition and Preprocessing
Utilizing the Pushshift API, 50,000 comments each from the "LabourUK" and "Tories" subreddits were acquired. The RedditDataAcquirer and RedditDataPreprocessor scripts handle data fetching and initial preprocessing, including tokenization and lemmatization, to clean and prepare data for analysis. When using the code for your own purposes this number can be easily tweaked in the configuration file. The pre-processing steps applied are shown below.

![image](https://github.com/Adam-R26/ReddiKnowSparse-Infer-Traits-of-Reddit-Users-from-Their-Comments/assets/53123097/038b59d3-d299-40ad-9d16-b75ab48a1e57)

##### Why Pre-Process the Data?
When we create bag of words based vectors, we have a column for every unique word that can appear within our documents. For large collections of documents this can be extremely high, its common to see feature vectors of around 20k different words. Of course most of these, words are not going to appear in an individual document meaning most rows within the training data contain zeros. When we have the case where most of the values in our vectors are zeros, we call them sparse vectors. A collection of sparse vectors making up our training data is then called a sparse matrix. 

Sparse matrices are problematic for machine learning models for a number of reasons. First of all, they increase the likelihood of overfitting. If we think about the amount of possibilities that occur in a matrix with 20000 columns, and assume we only have 1 or 0 in each row for whether or not a word occured and don't care about the amount of times the word occured the number of possibilities is 2^20000 (2 to the power of 20,000) this is an incomprehensively large number. Therefore without an incomprehensively large amount of training data to train our models on, most regions within this possiblity space will have extremely low coverage of no coverage. This means the model needs to make vast generalizations about new unseen data within these regions leading to poor performance. In other words in will need to massively overfit to the small amount of data it has...

Another issue is with the computational complexity required to train algorithms with a wide dataset of 20000 different columns. The larger the training data the more that data needs to be loaded into memory at a given time to update model parameters. Using a design matrix of this size you can run into problems very fast... To give a practical example lets consider the KNN algorithm, known as a lazy learner as it actually doesn't have a training process. Instead at run time the whole dataset is loaded into memory and a distance metric is used to compute the distance between an incoming sample and all of them in the training set. If the size of the dataset is n. Lets say that n is 100,000 samples. But every sample consists of 20,000 64 bit integers. This 0.00016 gigabytes per row. Multiplied by 100,000 this is 0.16 GB and we get 16 GB or memory for a single prediction. Now lets see, use this in a commerical setting where we make about 10 predictions a second... 160GB of RAM is needed. Of course this an extreme example but illustrates the point well.

Therefore our goal with pre-processing is to reduce our dimensionality as much as possible by removing any unnecesary noise from the text that might result in additional columns being created. For example, Fair and fair are the same word and should go in the same column but without case normalization there would be two columns. Similarly for punctuation, 'hello.' and and 'hello' should be the same. Lemmatization takes this a step further and reduces all words with the same inherent meaning to their base word. I.e., runs, running, ran would all be converted to run. Applying these steps reduces the high complexity of the solution sample space, and leads to less sparse vectors to work with, leading to better models. 

###  Feature Extraction (Vectorization)
The FeatureExtractor implements TF-IDF and count vectorization to transform textual data into a numerical format, enabling machine learning models to process and learn from the data effectively. A definition of count vectorization is below for your convienience:

#### Bag of Words (Count Vectorization): 
Represents every statement as a collection of words without considering order or structure. We only care about word frequencies in statements, paragraphs or documents, with each column representing the number of times a word is seen in each comment. One drawback of the bag of words model is the phenomenon known as Zipf’s law. Zipf’s law states that common words appear exponentially more times than less common words, meaning the bag is generally dominated by these words that aren’t as important for classification. 

To overcome Zipf'law the Term Frequency Inverse Document Frequency (TF-IDF) was created... But before we delve into that, have a look at the example below which shows the process of vectorization using bag of words as detailed above.

![image](https://github.com/Adam-R26/ReddiKnowSparse-Infer-Traits-of-Reddit-Users-from-Their-Comments/assets/53123097/5d457817-3850-41a7-b53f-e58a2d1fdda3)

#### Term Frequency Inverse Document Frequency:

The transformation that is applied to the bag of words model to statistically up weight words that occur less frequently within the collection of documents. This is used to combat the drawback of low information terms dominating the bag of words, by down weighting these words and up weighting less frequently occurring high information terms. The following formula is used to perform this:

TF-IDF = TF(t,d) * IDF(t)

Where: 
	TF(t, d) = Number of Occurrences of Term t in Document d / Number of Words in d
	IDF(t) = log(Total Amount of Documents/ Number of Documents that Contain Term t)
	
Analyzing the above formula, the term frequency, TF,  considers the number of times a specific term appears within a document. Whereas the inverse document frequency performs the up weight of this based on the scarcity of the term in the corpus. Logarithms are then applied to the inverse document frequency to dampen the effect of this up weight, as without it the up weight is often too strong in practice.


### Model Building and Training
A variety of machine learning models, from traditional algorithms like Random Forest, SVM, and Logistic Regression to advanced deep learning models including CNNs were applied. The process, orchestrated by the ModelBuilder script, involves extensive hyperparameter tuning (detailed in the HyperparameterGridConfigs script) to optimize model performance. Repeated Stratified K-Fold Cross Validation is used along with Grid Search, to search for the best combination of hyperparameters in an unbiased manor.  

### Evaluation and Visualization
Model performances were evaluated based on accuracy, precision, recall, and F1 score, with results visualized using matplotlib plots generated by the VisualizationPlotter. This comprehensive evaluation approach ensured a clear understanding of each model's effectiveness in sentiment analysis.

### Software Testing
Units tests were produced to test the overall functionality of the pipeline, as the pipeline was only intended for personal use. In a production setting, much more comprehensive integration, unit and smoke tests would be needed, if we want wanted to use the pipeline to produce new models on a regular basis.

### Key Findings and Contributions
#### TF-IDF Transformation: Significantly improves model performance, especially for KNN and SVM classifiers.
![image](https://github.com/Adam-R26/ReddiKnowSparse-Infer-Traits-of-Reddit-Users-from-Their-Comments/assets/53123097/db206847-ab85-4163-b4e7-2eba89a7241c)


#### Preprocessing Impact: Universal improvement across all models, underlining the importance of thorough data cleaning and preparation.
![image](https://github.com/Adam-R26/ReddiKnowSparse-Infer-Traits-of-Reddit-Users-from-Their-Comments/assets/53123097/608387a6-3ab0-4fde-a6f1-f76c587eb2d9)

#### Other Notes
- KNN for Text Classification: Found to be less effective compared to other models, particularly before applying TF-IDF and preprocessing steps. This is a general observation for the KNN algortithm, in that in struggles to handle datasets of high dimensionality due to the so called 'Curse of Dimensionality'. Essentially the larger the number of dimensions the larger chance of two samples being close in Euclidean space, simply due to chance rather than them being related by class, which reduces the effectivenes of KNN.

### Technologies and Libraries
This project leverages Python for its implementation, with extensive use of libraries such as Pandas for data manipulation, scikit-learn for machine learning, TensorFlow for deep learning models, and matplotlib for data visualization.

### Future Directions
Future work will explore the application of more sophisticated neural network architectures, inclusion of larger and more diverse datasets, and further refinement of preprocessing and feature extraction techniques to enhance model accuracy and reliability.

### Conclusion
- This project offers a valuable introduction to the world of applied NLP. Allowing users to easily demonstrate the effectivness of pre-preprocessing on text data, as well as term frequency inverse document frequency, which can be used to enhance traditional sparse vectors. Note it is meant to be an educational piece only introducing users to how NLP can be applied effectively.
- It leverages a simple heuristic approach to create a dataset that can be used to solve a real world problem. More modern techniques could involve manually labelling a small set of data and using an LLM to synthetically generate more data based upon these examples. In this way we could achieve a much cleaner dataset, to train our models on.
- Using such a dataset we would expect to see a larger difference between simpler models such as logistic regression and the deep learning architectures due to their ability to fit higher variance decision boundaries. In this instance with the current noisy dataset, the models all have similar performance, as the more complex models are overfit to the data harming generalization performance.
- The project could be improved by expanding to multi-class classification, going into a more in depth evaluation using an Reciever Operating Characteristic (ROC) curve to understand the trade-off between Recall (TPR) and False Alarm Rate (FPR). This curve would allow the models to be prepared at all different thresholds, which would be vital in a commerical setting to understanding where to set the threshold.
- It could also be improved by improving the testing adding in more comprehensive unit testing, integration tests and if hosted on a cloud platform smoke tests.
- Finally it could be improved using dense vector representations from tranformers, or pre-training transformer based neural architectures for classification. 
