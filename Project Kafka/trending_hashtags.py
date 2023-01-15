#Using the river-library to do sentiment analysis

from kafka import KafkaConsumer, KafkaProducer, TopicPartition
from transformers import pipeline
import numpy as np
import pandas as pd
import unicodedata
import nltk
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from river import feature_extraction
import re

# Define Kafka consumer and topic to consume from
consumer = KafkaConsumer(bootstrap_servers='localhost:9092')
topicName = "tweets-labeled-demo"

# Getting the last offset of the messages
tp = TopicPartition(topicName,0)
consumer.assign([tp])
consumer.seek_to_end(tp)
lastoffset = consumer.position(tp)
consumer.seek_to_beginning(tp)

# Consume the messages
X = []
y = []
for msg in consumer:
    tweet = msg.value[4:].decode()
    X.append(tweet)
    label = msg.value[0:3].decode() == ("POS" or "NEU")
    y.append(label)
    if (msg.offset == lastoffset - 1):
        break

#Turn the messages and their labels to a dataframe
full_dataset = np.concatenate((np.array(X).reshape(-1,1),np.array(y).reshape(-1,1)),axis = 1)
df_tweets = pd.DataFrame(full_dataset,columns = ['tweet','label'])
print(df_tweets.head())

# Do some cleaning on the tweets
#nltk.download('stopwords')
stp_list = stopwords.words('english') + stopwords.words('french')
def clean_text(text1):
    rt = text1[0:3]
    if (rt == "RT "):
        text1 = re.sub('RT @.*:','',text1)
    text1 = text1.lower()
    text1 = unicodedata.normalize('NFKD',text1).encode('ascii', errors='ignore').decode('utf-8')
    text1 =  re.sub('[^A-Za-z]',' ',text1)
    text1 = re.sub(r'\b\w{1,3}\b', '', text1)
    text1 = re.sub(r'\s+', ' ',text1)
    text1 = ' '.join([word for word in text1.split() if word not in stp_list])
    return text1

# Transform the data using tfidf
X_train, X_test, y_train, y_test = train_test_split(df_tweets['tweet'], df_tweets['label'], test_size=0.33, random_state=42)
cv1 = TfidfVectorizer(lowercase=True, min_df=3,  max_features=None,
                      strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                      ngram_range=(1, 3))
vectorized1 = cv1.fit_transform(X_train)
vectorized2 = cv1.transform(X_test)

# Find hashtags
def find_hashtags(tweet):
    return re.findall('(#[A-Za-z]+[A-Za-z0-9-_]+)', tweet)
hashtags = df_tweets.iloc[:,0].apply(lambda x: find_hashtags(x))
hashtags = [item for sublist in np.unique(hashtags.array) for item in sublist] #convert list of lists to a flat_list

# Convert the hashtags to lower format
hashtags = [word.lower() for word in hashtags]

# Create a dictionary of hashtags with their specific count
hashtags_dict = {}
for hashtag in hashtags:
    if (hashtag in hashtags_dict.keys()):
        hashtags_dict[hashtag] +=1
    else:
        hashtags_dict[hashtag] = 1

# Print the ranking for the used hashtags :
sorted_hashtags = sorted(hashtags_dict.items(), key=lambda x:x[1] , reverse= True)

# Print the top 10 hashtags
print(sorted_hashtags[:10])




