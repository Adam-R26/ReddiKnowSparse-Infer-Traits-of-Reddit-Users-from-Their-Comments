#Import required libraries
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

class FeatureExtractor:
    def __init__(self, further_pre_process, tf_idf, min_df=1, max_df=1.0, test_size=0.2, max_features=None, vectorizer=False, ngram_range=(1,1), stop_words='english'):
        self._further_pre_process = further_pre_process
        self._tf_idf = tf_idf
        self._min_df = min_df
        self._max_df = max_df
        self._test_size = test_size
        self._max_features = max_features
        self._ngram_range = ngram_range
        self._stop_words=stop_words
        self.vectorizer = vectorizer
        

    def extract_bow_features(self, df) -> list:
        '''Extracts features from text samples, based on the pipeline configuration'''
        #Split into train and test
        Y = df['Target'].copy()
        X = df.drop('Target', axis=1).copy()
        
        #Random state set for reproducible results.
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=self._test_size, random_state=123, stratify=Y)
        
        #Fit vectorizer to the vocabulary within our data.
        self.vectorizer = self._vectorizer_factory()
        self.vectorizer.fit(X_train['body'])
        
        X_train = self.vectorizer.transform(X_train['body'])
        X_test = self.vectorizer.transform(X_test['body'])
        
        return X_train, X_test, Y_train, Y_test, len(self.vectorizer.vocabulary_)
    
    def _vectorizer_factory(self) -> None:
        '''Sets appropriate vectorized based on run-time paramters'''
        if self._tf_idf == True and self._further_pre_process == True:
            return TfidfVectorizer(ngram_range=self._ngram_range , max_features=self._max_features, binary=False, stop_words=self._stop_words, max_df=self._max_df, min_df=self._min_df)
        
        elif self._tf_idf == False and self._further_pre_process == True:
            return CountVectorizer(ngram_range=self._ngram_range, max_features = self._max_features, binary=False, stop_words=self._stop_words, max_df=self._max_df, min_df=self._min_df)
            
        elif self._tf_idf == True and self._further_pre_process == False:
            return TfidfVectorizer(ngram_range=self._ngram_range , max_features=self._max_features, binary=False, stop_words=None, max_df=self._max_df, min_df=self._min_df, lowercase=False, strip_accents=None, token_pattern=r'[a-zA-Z0-9$&+,:;=?@#|<>.^*()%!-]+')
            
        elif self._tf_idf == False and self._further_pre_process == False:
            return CountVectorizer(ngram_range=self._ngram_range , max_features=self._max_features, binary=False, stop_words=None, max_df=self._max_df, min_df=self._min_df, lowercase=False, strip_accents=None, token_pattern=r'[a-zA-Z0-9$&+,:;=?@#|<>.^*()%!-]+')
        else:
            raise("Invalid tf_idf or pre_process paramter value.")
            
    

            