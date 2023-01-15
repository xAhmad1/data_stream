# This code builds an online tweet sentiment analysis using the river library

# Imports
from kafka import KafkaConsumer, TopicPartition
from river import linear_model
from river import feature_extraction
from river import optim
import re
import unicodedata
from nltk.corpus import stopwords

# Defining the tfidf feature extractor
tfidf = feature_extraction.TFIDF()

# Defining the consumer and topic to consume from
consumer = KafkaConsumer(bootstrap_servers='localhost:9092')
topicName = "tweets-labeled-demo"


# Defining the model
model = linear_model.LogisticRegression()


# Getting the size of the topic
tp = TopicPartition(topicName,0)
consumer.assign([tp])
consumer.seek_to_end(tp)
lastOffset = consumer.position(tp)
consumer.seek_to_beginning(tp)
LR = linear_model.LogisticRegression(optimizer=optim.SGD(.1))

# Defining a function to clean text read from the topic
stp_list = stopwords.words('english') + stopwords.words('french')
def clean_text(text1):
    rt = text1[0:3]
    if (rt == "RT "):
        text1 = re.sub('RT @.*:','',text1)
    text1 = text1.lower()
    text1 = unicodedata.normalize('NFKD',text1).encode('ascii', errors='ignore').decode('utf-8')
    text1 =  re.sub('[^A-Za-z]',' ',text1)
    text1 = re.sub(r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', ' ',text1)
    text1 = re.sub('https', ' ',text1)
    text1 = re.sub(r'\s+', ' ',text1)
    text1 = ' '.join([word for word in text1.split() if word not in stp_list])
    return text1

# Looping on each msg in the topic, updating the tfidf matrix then updating our model
for msg in consumer:
    sentence = (msg.value[4:]).decode()
    sentence = clean_text(sentence)
    label = (msg.value[0:3]).decode()==("POS" or "NEU")


    tfidf = tfidf.learn_one(sentence)
    LR = LR.learn_one(tfidf.transform_one(sentence),label)
    

    if(msg.offset == lastOffset-1):
        print(lastOffset)
        accuracy = (1 - sum/lastOffset) * 100
        print("accuracy = {}% ".format(accuracy))
        break


