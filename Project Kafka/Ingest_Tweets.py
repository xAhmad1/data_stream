# this code will ingest tweets into a kafka topic called raw-tweets

import tweepy
import kafka
from kafka import KafkaProducer
import re

client = tweepy.Client(bearer_token='AAAAAAAAAAAAAAAAAAAAALv2kgEAAAAAAFPj4lftyFcSuE9yxVAufMmgkPM%3DbYJcYlZUJJjAPk5m7yd3S51B4tEjYQL4F5CcTX56DF0tg8jnVb')

# Replace with your own search query

query = 'France'
producer = KafkaProducer(bootstrap_servers='localhost:9092')
topic_name = 'raw-tweets-demo'

size = 100
for tweet in tweepy.Paginator(client.search_recent_tweets, query=query,
tweet_fields=[ 'created_at', 'lang'], max_results=size).flatten(limit=3000):
    data = "\n" + tweet['lang'] + " \n" + tweet['text']
    producer.send(topic_name, str.encode(data))
