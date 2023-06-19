from django.core.management.base import BaseCommand
import os
import sys
import csv
# from django.conf import settings
import snscrape.modules.twitter as sntwitter
import snscrape
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from main.models import PidginDB, SentiDB, QueryDB
import wordcloud
# import matplotlib.pyplot as plt
import numpy as np
from textblob import TextBlob
from IPython.display import display
from collections import defaultdict

# import matplotlib.pyplot as plt 
import re
import string
from django.forms.models import model_to_dict

import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import words
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
nltk.download('stopwords')
nltk.download('vader_lexicon')

from collections import Counter
from googletrans import Translator

# from matplotlib import pyplot as plt
# from matplotlib import ticker
# import seaborn as sns
# import plotly.express as px
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)



class Command(BaseCommand):
    # def add_arguments(self, parser):
    #     parser.add_argument('q', type=str)
    #     # parser.append(min(opt.option_strings))
    #     parser.add_argument('args', nargs='*')
    help = "scrape tweets and analyze tweets"
    def handle(self, *args, **options):
        # q=options.get('q', None)
        # os.system("snscrape --jsonl --max-results 100 --since 2022-01-01 twitter-search 'desmond elliot until:2020-10-31' > text-query-tweets.json")
        # os.system("snscrape --jsonl --max-results 51 --since 2022-01-01 twitter-search 'desmond elliot' > text-query-tweets.json")
        # Creating list to append tweet data to
        tweets_list2 = []
        get_q = QueryDB.objects.all().order_by('-created')[0]
        # print(q)

        # # Using TwitterSearchScraper to scrape data and append tweets to list
        # for i,tweet in enumerate(sntwitter.TwitterSearchScraper('Desmond Elliot since:2020-10-20 until:2020-10-31').get_items()):
        for i,tweet in enumerate(sntwitter.TwitterSearchScraper(str(get_q.keyword)+' '+'since:2022-01-01').get_items()):

            if i>60:
                break
            tweets_list2.append([tweet.date, tweet.id, tweet.content, tweet.user.username])
            
        # # Creating a dataframe from the tweets list above
        df = pd.DataFrame(tweets_list2, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])
        df.to_csv('tweets.csv') 
        df = pd.read_csv('tweets.csv', encoding='latin-1')
        
        #check for duplicates
        # df.duplicated().sum()
        
        #check for size of dataset
        df.shape

        df.head(50)

        df.describe()

        # Create stopword list:
        from wordcloud import WordCloud, STOPWORDS
        stopwords = set(STOPWORDS)

        # pull out needed columns
        new_df = df[['Datetime', 'Text','Username']]
        

        # newhead = df.head()
        df = df.head(50)

        df.info()

        df.Text = df.Text.astype('str')


        def cleaner(tweet):
            # clean tweets
            tweet = re.sub("@[A-Za-z0-9]+","",tweet) #Remove @ sign
            tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet) #Remove http links
            tweet = " ".join(tweet.split())
            tweet = tweet.replace("#", "").replace("_", " ") #Remove hashtag sign but keep the text
            return tweet
        df['Text'] = df['Text'].map(lambda x: cleaner(x))

        # get all pidgin objects
        pidgin_dict = PidginDB.objects.all()
        # new_dict=[]
        mn1 = {}
        for obj in pidgin_dict:
            mn = {str(obj.pidgin.lower()):str(obj.english.lower()),};
            mn1.update(mn)
        translator = Translator()
        new_corpus = []
        for text in df['Text'][:50]:
            new_text = text.lower()
            res = " ".join(mn1.get(ele, ele) for ele in new_text.split())
            # translate from Igbo to English
            igbo2eng = translator.translate(str(res), dest='en')
            new_corpus.append(igbo2eng.text)
        
        df2 = pd.DataFrame(new_corpus, columns=['Text'])
        # # # print(df2)
        df['Text'] = df2['Text'].values
        analyzer =  SentimentIntensityAnalyzer()
        df['compound'] = [analyzer.polarity_scores(x)['compound'] for x in df['Text']]
        df['neg'] = [analyzer.polarity_scores(x)['neg'] for x in df['Text']]
        df['neu'] = [analyzer.polarity_scores(x)['neu'] for x in df['Text']]
        df['pos'] = [analyzer.polarity_scores(x)['pos'] for x in df['Text']]

        # df.head()
        #labelize tweet
        label = lambda x:'neutral' if x==0 else('positive' if x>0 else 'negative')
        df['label'] = df.compound.apply(label)
        # display(df.head(10))
        senti = df.label.value_counts()
        
        df.loc[df['label'] == 'negative', 'cyberbully'] = df['Username']
        df.loc[df['label'] == 'negative', 'original_tweet'] = new_df['Text']
        
        bullies = []
        for i in df['cyberbully']:
            bullies.append(i)
        
        clean_df = df[['original_tweet', 'label','Username']]
        df = clean_df
        for index, row in df.iterrows():
            try:
                get_senti = SentiDB.objects.get(tweet=row['original_tweet'])
            except (SentiDB.DoesNotExist):
                new_data = SentiDB(tweet=row['original_tweet'], sentiment=row['label'], username=row['Username'])
                new_data.save()
                print('saved')
            

