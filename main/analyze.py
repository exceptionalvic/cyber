import os
import sys
import csv
# from django.conf import settings
from models import PidginDB

# # Using OS library to call CLI commands in Python
# os.system("snscrape --jsonl --max-results 500 --since 2020-06-01 twitter-search 'its the elephant until:2020-07-31' > text-query-tweets.json")


with open('dict.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # The header row values become your keys
        pidgin = row['pidgin']
        english = row['english']
        # etc....

        new_dict = PidginDB(pidgin=pidgin, english=english)
        new_dict.save()
        print('saved')