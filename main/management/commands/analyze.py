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
# sns.set(style="darkgrid")

# # Using OS library to call CLI commands in Python
# os.system("snscrape --jsonl --max-results 500 --since 2020-06-01 twitter-search 'its the elephant until:2020-07-31' > text-query-tweets.json")

# class Command(BaseCommand):
#     help = "collect Tweets"
#     def handle(self, *args, **options):
#         with open('dict.csv') as csvfile:
#             reader = csv.DictReader(csvfile)
#             for row in reader:
#                 # The header row values become your keys
#                 pidgin = row['pidgin']
#                 english = row['english']
#                 # etc....

#                 new_dict = PidginDB(pidgin=pidgin, english=english)
#                 new_dict.save()
#                 print('saved')


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
        # df.head()
        # pidgin_dict = PidginDB.objects.all()
        # model_to_dict=[model for model in pidgin_dict.values()]
        # # print(model_to_dict)
        # res = " ".join(model_to_dict.get(ele, ele) for ele in df['Text'])
        # print("Replaced Strings : " + str(res))
        # pidgin_dict1 = model_to_dict(pidgin_dict)

        # res = " ".join(model_to_dict.get(ele, ele) for ele in all_words.split())

        # test_str = 'Osinbajo, soro soke werey. Buhari don die'
        # pidgin_dict = PidginDB.objects.all()
        # # new_dict=[]
        # mn1 = {}
        # for obj in pidgin_dict:
        #     mn = {str(obj.pidgin.lower()).replace(" ",""):str(obj.english.lower()).replace(" ",""),};
        #     mn1.update(mn)
            # merged = {**mn}
            # new_dict.append(mn)
            # mn = obj.pidgin
        # print(mn1)
        # super_dict = defaultdict(set)  # uses set to avoid duplicates
        # for d in new_dict:
        #     for k, v in d.items():  # use d.iteritems() in python 2
        #         super_dict[k].add(v)
        # super_dict = {}.replace("[","]","")
        # for k in set(k for d in new_dict for k in d):
        #     super_dict[k] = [d[k] for d in new_dict if k in d]
        # print(super_dict)
        # res = " ".join(super_dict.get(ele, ele) for ele in df['Text'][:5])
        # res = " ".join(mn1.get(ele, ele) for ele in test_str.split())
          
        # # # printing result 
        # print("Replaced Strings : " + str(res))

            # mn[str(obj.pidgin)].append(str(obj.english))
            # print(mn)

        # my_dict = {"Name":[],"Address":[],"Age":[]};

        # my_dict["Name"].append("Guru")
        # my_dict["Address"].append("Mumbai")
        # my_dict["Age"].append(30)   
        # print(my_dict)

        # new_dict = model_to_dict(pidgin_dict)
        # print(pidgin_dict)
        # model_to_dict=[model for model in pidgin_dict.values()]

        # pidgin_dict = {'ode':'fool', 'na':'is','craze':'crazy','were':'mad', 'abeg':'please', 'fly':'fly'}

        # res = " ".join(pidgin_dict.get(ele, ele) for ele in test_str.split())
        # res = " ".join(pidgin_dict.get(ele, ele) for ele in test_str)
          
        # printing result 
        # print("Replaced Strings : " + str(res))

        # newhead = df.head()
        df = df.head(50)

        df.info()

        df.Text = df.Text.astype('str')


        def cleaner(tweet):
            tweet = re.sub("@[A-Za-z0-9]+","",tweet) #Remove @ sign
            tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet) #Remove http links
            tweet = " ".join(tweet.split())
            #tweet = ''.join(c for c in tweet if c not in emoji.UNICODE_EMOJI['en']) #Remove Emojis
            tweet = tweet.replace("#", "").replace("_", " ") #Remove hashtag sign but keep the text
            return tweet
        df['Text'] = df['Text'].map(lambda x: cleaner(x))

        # all_words = ' '.join([text for text in df['Text']])
        # test_str = 'Osinbajo, soro soke werey. Buhari don die'
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
            # print("Main Text: "+new_text)
            # print("")cd..splitcd..replacecd
            # translate from pidgin to english using our custom pidgin dictionary
            # res = " ".join(mn1.get(ele, ele) for ele in igbo2eng.split())
            # res = " ".join(mn1.get(ele, ele) for ele in igbo2eng)
            # print(igbo2eng.text)
            # df.replace(str(df['Text']), str(igbo2eng.text), inplace=True)
        # new_df1 = df[['Datetime', 'Text','Username']]

            new_corpus.append(igbo2eng.text)
        # print(new_corpus)
        # trunc_corpus = new_corpus
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
        # print(df)
        # print(senti)
        df.loc[df['label'] == 'negative', 'cyberbully'] = df['Username']
        df.loc[df['label'] == 'negative', 'original_tweet'] = new_df['Text']
        # df.loc[df['label'] == 'positive', 'original_tweet'] = new_df['Text']
        # df.loc[df['label'] == 'neutral', 'original_tweet'] = new_df['Text']

        # df['Datetime'] = new_df['Datetime']
        # print(df['cyberbully'])
        bullies = []
        for i in df['cyberbully']:
            bullies.append(i)
        # print(bullies)
        # print(df)
        clean_df = df[['original_tweet', 'label','Username']]
        df = clean_df
        # new_dict = df.to_dict()
        # new_dict1 = df.to_json()
        # print(df)
        for index, row in df.iterrows():
            try:
                get_senti = SentiDB.objects.get(tweet=row['original_tweet'])
            except (SentiDB.DoesNotExist):
                new_data = SentiDB(tweet=row['original_tweet'], sentiment=row['label'], username=row['Username'])
                new_data.save()
                print('saved')
            # print (index,row["label"], row["Username"])
        # SentiDB.objects.all().delete()
        # final_array = {}
        # for t in new_dict:
        #     fn = {str(t.Text):str(t.Username.lower()),};
            # final_array.update(fn)
            # new_data = SentiDB(tweet=t['original_tweet'])
        #     # username = t['Username'].value
        #     # tweet = t['original_tweet']
        #     new_data = SentiDB(tweet=t['Username'], english=english)
        #     # new_dict.save()
            # print(t)


        # for t in df['Text']:
        #     final_df = new_df.replace(str(df['Text']), str(igbo2eng.text), inplace=True)
        # print(final_df)
        # df['Text'] = df2.replace(df['Text'], df['Text'])
        # print(new_df)
        # print(df)
      
             # # printing result 
            # print("Replaced Strings : " + str(res))
        # pidgin_dict = PidginDB.objects.all()
        # model_to_dict=[model for model in pidgin_dict.values()]
        # # pidgin_dict1 = model_to_dict(pidgin_dict)

        # # res = " ".join(model_to_dict.get(ele, ele) for ele in all_words.split())

        # print("Replaced Strings : " + str(res))
        # print(all_words)

        # from wordcloud import WordCloud
        # wordcloud = WordCloud(background_color = 'white', width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

        # plt.figure( figsize=(20,10) )
        # plt.imshow(wordcloud)
        # plt.axis("off")
        # plt.show()
        # print(df['Text'])

