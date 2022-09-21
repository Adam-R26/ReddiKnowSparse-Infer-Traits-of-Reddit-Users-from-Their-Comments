#Import required libraries
import re
import string
import nltk
from nltk import word_tokenize
from nltk import pos_tag 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import pandas as pd


class RedditDataPreprocessor:    
    def __init__(self, pre_process_data:bool, class_data_points:int):
        self._pre_process_data = pre_process_data
        self._class_data_points = class_data_points
        
    def preprocess_data(self, df:pd.DataFrame) -> pd.DataFrame:
        '''Applies pre-processing methods on the data if required'''
        #Remove nonsensical comments
        df = self._prune_data_by_score(df)
        
        #Perform additional pre-processing if required.
        if self._pre_process_data and self._class_data_points>0:
            #Clean data - Removing punctuation, email address', numbers, extra white spaces etc.
            df['body'] = df['body'].apply(lambda x:self._clean_data(x))
    
            #Lemmatize data - Use one coherent word to represent each meaning within the comments.
            df = self._lemmatize_data(df)
            
        elif self._pre_process_data:
            raise ValueError("Number of Comments Per Class Parameter Must be More than 0.")
            
        #Remove any comments that are empty
        df = df.dropna()
                
        return df
    
    def _prune_data_by_score(self, df:pd.DataFrame) -> pd.DataFrame:
        '''Retrieves the top n comments ordered by score for each class based on class_data_points parameter.'''
        #Separate into distinct classes
        df_c1 = df.loc[df['Target'] == 0].copy()
        df_c1 = df_c1.sort_values('score', ascending=False)
        df_c1 = df_c1[0:self._class_data_points]
        
        df_c2 = df.loc[df['Target'] == 1].copy()
        df_c2 = df_c2.sort_values('score', ascending=False)
        df_c2 = df_c2[0:self._class_data_points]
        
        output = pd.concat([df_c1, df_c2])
        
        return output
        
    def _clean_data(self, text: str) -> str:
        '''Function pre-processes text into a form that can go into the lemmatization step.'''
        text = text.lower()
        
        #Remove punctuation
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        
        #Remove words with numbers in them
        text = re.sub('\w*\d\w*', '', text)
        
        #Remove email addresses'
        text = re.sub('\w*@\w*', '', text)
        
        #Remove numbers
        text = re.sub('\d', '', text)
        
        #Remove special characters
        text = re.sub('[^A-Za-z0-9]+', ' ', text)
        
        return text
    
    def _get_wordnet_pos(self, word: str) -> str:
        '''Map part-of-speech tag to characters lemmatize() accepts'''
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)
    
    def _lemmatize_data(self, df:pd.DataFrame) -> pd.DataFrame:
        '''Applies functions for tokenization, pos tagging, stopword removal before lemmatization.'''
        #Tokenize the words within the sentences
        df['body'] = df['body'].apply(lambda x: word_tokenize(x))
    
        #Get part of speech for each word within each comment
        df['body'] = df['body'].apply(lambda x: pos_tag(x))
    
        #Peform lemmatization of each comment based on pos
        lemmatizer = WordNetLemmatizer()
        df['body'] = df['body'].apply(lambda x: [lemmatizer.lemmatize(i[0], self._get_wordnet_pos(i[1])) for i in x])
    
        #Remove stop words from the comments
        stopwords_english = stopwords.words('english')
        df['body'] = df['body'].apply(lambda x: [i for i in x if i not in stopwords_english])
        
        #Combine words in together in list
        df['body'] = df['body'].apply(lambda x: ' '.join(x))
        return df
    