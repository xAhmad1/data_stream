#Using the river-library to do sentiment analysis

from kafka import KafkaConsumer, KafkaProducer, TopicPartition
import numpy as np
import pandas as pd
import unicodedata
from river.cluster import STREAMKMeans
from river import compose
from river import feature_extraction
from river import cluster
import re

# Define Kafka consumer and topic to consume from
consumer = KafkaConsumer(bootstrap_servers='localhost:9092')
topicName = "tweets-labeled"

# Getting the last offset of the messages
tp = TopicPartition(topicName,0)
consumer.assign([tp])
consumer.seek_to_end(tp)
lastoffset = consumer.position(tp)
consumer.seek_to_beginning(tp)

# Define the text cleaning functoin
#stp_list = stopwords.words('english') + stopwords.words('french')
def clean_text(text1):
    rt = text1[0:3]
    if (rt == "RT "):
        text1 = re.sub('RT @.*:','',text1)
    text1 = text1.lower()
    text1 = unicodedata.normalize('NFKD',text1).encode('ascii', errors='ignore').decode('utf-8')
    text1 =  re.sub('[^A-Za-z]',' ',text1)
    text1 = re.sub(r'\b\w{1,3}\b', '', text1)
    text1 = re.sub(r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', ' ',text1)
    text1 = re.sub('https', ' ',text1)
    text1 = re.sub(r'\s+', ' ',text1)
    #text1 = ' '.join([word for word in text1.split() if word not in stp_list])
    return text1

# Consume the messages

for msg in consumer:
    tweet = msg.value[4:].decode()
    label = msg.value[0:3].decode() == ("POS" or "NEU")

    #y_pred = model.predict_one(tweet)
    tweet = clean_text(tweet)
    tweet_bow = feature_extraction.BagOfWords(lowercase=True, ngram_range=(1, 1)).transform_one(tweet)
    clusterer = STREAMKMeans(chunk_size = 1 , n_clusters= 4)
    clusterer = clusterer.learn_one(tweet_bow)
    print(clusterer.predict_one(tweet_bow))
    #model = model.learn_one(tweet)

    if (msg.offset == lastoffset - 1):
        break

print(clusterer.centers)
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA

# # Set up the PCA model to reduce the dimensionality of the data
# pca = PCA(n_components=2)

# # Perform PCA on the cluster centroids
# centroids = clusterer.centers
# #print("ok")
# #print(list(centroids.values()))
# #print("ok")
# reduced_centroids = pca.fit_transform(centroids)

# # Plot the clusters
# plt.scatter(reduced_centroids[:,0], reduced_centroids[:,1])
# plt.show()


