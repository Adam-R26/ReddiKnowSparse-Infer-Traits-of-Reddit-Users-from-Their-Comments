import pandas as pd
import datetime as dt
from pmaw import PushshiftAPI

class RedditDataAcquirer:
    def __init__(self, before_date: str, after_date: str, subreddits: list, number_of_comments: int, data_file_path: str):    
        before_date_split = before_date.split("-")
        after_date_split = after_date.split("-")
        self._before_date = int(dt.datetime(int(before_date_split[0]), int(before_date_split[1]), int(before_date_split[2]) ,0 ,0).timestamp())
        self._after_date = int(dt.datetime(int(after_date_split[0]), int(after_date_split[1]), int(after_date_split[2]) ,0 ,0).timestamp())
        self._subreddits = subreddits
        self._number_of_comments = number_of_comments
        self._data_file_path = data_file_path
        
    def acquire_data(self, use_api_flag: bool) -> pd.DataFrame():
        '''Acquires the data using the API or from memory depending on the "use_api_flag'''
        if use_api_flag and self._number_of_comments>0:
            subreddit_data = []
            for i in range(len(self._subreddits)):
                df_tmp = self._get_subreddit_comments_pmaw(self._subreddits[i])
                df_tmp['Target'] = i
                subreddit_data.append(df_tmp)
            
            df = pd.concat(subreddit_data)
            
        elif use_api_flag and self._number_of_comments<=0:
            raise ValueError("Number of Comments Per Class Parameter Must be More than 0.")
            
        else:
            df = pd.read_csv(self._data_file_path)
            
        return df
    
    def _get_subreddit_comments_pmaw(self, subreddit: str) -> pd.DataFrame():
        '''Retrieves data from subreddit and returns in dataframe'''
        api = PushshiftAPI()
        submissions = api.search_comments(subreddit=subreddit, limit=self._number_of_comments, before=self._before_date, after=self._after_date)
        sub_df = pd.DataFrame(submissions)
        sub_df = sub_df[['id', 'body', 'score', 'author']]
        return sub_df
    
    
    

    
    
    
    
    
    
        
        
    
    
    
