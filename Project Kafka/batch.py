#Using the river-library to do sentiment analysis

from kafka import KafkaConsumer, KafkaProducer, TopicPartition
from transformers import pipeline
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from river import feature_extraction
import re
# Define a function to clean the data:
def clean_text(df, text_field):
    df[text_field] = df[text_field].str.lower()
    df[text_field] = df[text_field].apply(lambda elem: re.sub(r"(@[A-Za-z0–9]+)|([⁰-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "",   elem)) 
    return df


sent_pipe = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")
consumer = KafkaConsumer(bootstrap_servers='localhost:9092')
producer = KafkaProducer(bootstrap_servers='localhost:9092')
topicName = "tweets-labeled11"
# Read and print message from consumer

tp = TopicPartition(topicName,0)
consumer.assign([tp])

#Getting the size of the topic
consumer.seek_to_end(tp)
lastOffset = consumer.position(tp)
consumer.seek_to_beginning(tp)
sum = 0
X = []
y =[]
for msg in consumer:
    sentence = (msg.value[4:]).decode()
    X.append(sentence)
    label = (msg.value[0:3]).decode()==("POS" or "NEU")
    y.append(label)
    if(msg.offset == lastOffset-1):
        print(lastOffset)
        break

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
cv1 = TfidfVectorizer(lowercase=True, min_df=3,  max_features=None,
                      strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                      ngram_range=(1, 3))
                      #stop_words= stop_wordss)

vectorized1 = cv1.fit_transform(X_train)
vectorized2 = cv1.transform(X_test)
clf = LogisticRegression(random_state=0).fit(vectorized1, y_train)
print(clf.score(vectorized2,y_test))