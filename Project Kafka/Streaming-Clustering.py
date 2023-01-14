#Using the river-library to do sentiment analysis

from collections import defaultdict
from kafka import KafkaConsumer, TopicPartition
import numpy as np
import unicodedata
from river.cluster import STREAMKMeans
from river import feature_extraction
import re
from collections import Counter
from nltk.corpus import stopwords

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
stp_list = stopwords.words('english') + stopwords.words('french')
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
    text1 = ' '.join([word for word in text1.split() if word not in stp_list])
    return text1

# Consume the messages
dict1 = {}
clusters_tweets = []
tweet_bow_list = []
nb_clusters = 3

clusterer = STREAMKMeans(chunk_size = 3 , n_clusters= nb_clusters)
for msg in consumer:
    
    tweet = msg.value[4:].decode()
    label = msg.value[0:3].decode() == ("POS" or "NEU")

    #y_pred = model.predict_one(tweet)
    tweet = clean_text(tweet)
    tweet_bow = feature_extraction.BagOfWords(lowercase=True, ngram_range=(1, 1)).transform_one(tweet)
    tweet_bow_list.append(tweet_bow)

    # reset the dictionary
    for word in dict1.keys():
        dict1[word] = 0

    # update the dictionary with the new values
    for word in tweet_bow:
        if word not in dict1.keys():
            dict1[word] = 1
        else:
            dict1[word] += 1
        
    
    clusterer = clusterer.learn_one(dict1)
    clusters_tweets.append(clusterer.predict_one(dict1))
    if (msg.offset == lastoffset - 1):
        break


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Set up the PCA model to reduce the dimensionality of the data
pca = PCA(n_components=2)
# Perform PCA on the cluster centroids
centroids = clusterer.centers


centroids_coordinates = np.zeros((nb_clusters,len(dict1)))
i = 0
j = 0
for vals in clusterer.centers.values():
    for k,val in vals.items():
        centroids_coordinates[i,j] = val
        j += 1
    j = 0
    i += 1

reduced_centroids = pca.fit_transform(centroids_coordinates)

# Creating coordinate for each tweet
coordinates_tweets = []
for tweet in tweet_bow_list:
    dict1 = dict1.fromkeys(dict1, 0)
    for word in tweet:
        dict1[word] += 1
    coordinates_tweets.append(list(dict1.values()))

coordinates_tweets = np.array(coordinates_tweets)
reduced_coordinates_tweets = pca.transform(coordinates_tweets)


for j in range(nb_clusters):
    indices = [i for i, x in enumerate(clusters_tweets) if x == j]
    print("Cluster {} words: ".format(j+1))

    #make a new dictionary:
    dict1 = Counter()
    for idx in indices:
        dict1 +=tweet_bow_list[idx]
    print(dict1) 
    print("\n\n")



# Plot the clusters
import matplotlib.cm as cm
plt.scatter(reduced_centroids[:,0], reduced_centroids[:,1])
plt.scatter(reduced_coordinates_tweets[:,0],reduced_coordinates_tweets[:,1],color = cm.hot(clusters_tweets))
plt.show()