#Using the river-library to do sentiment analysis

from kafka import KafkaConsumer, KafkaProducer, TopicPartition
from transformers import pipeline
from river import stream
from river import preprocessing
from river import linear_model
from river import feature_extraction
from river import optim
import re
tfidf = feature_extraction.TFIDF()


sent_pipe = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")
consumer = KafkaConsumer(bootstrap_servers='localhost:9092')
producer = KafkaProducer(bootstrap_servers='localhost:9092')

topicName = "tweets-labeled"
# Read and print message from consumer


model = linear_model.LogisticRegression()
tp = TopicPartition(topicName,0)
consumer.assign([tp])

#Getting the size of the topic
consumer.seek_to_end(tp)
lastOffset = consumer.position(tp)
consumer.seek_to_beginning(tp)
LR = linear_model.LogisticRegression(optimizer=optim.SGD(.1))
sum = 0

def clean_text(df, text_field):
    df[text_field] = df[text_field].str.lower()
    df[text_field] = df[text_field].apply(lambda elem: re.sub(r"(@[A-Za-z0–9]+)|([⁰-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "",   elem)) 
    return df


for msg in consumer:
    sentence = (msg.value[4:]).decode()
    label = (msg.value[0:3]).decode()==("POS" or "NEU")


    #print(LR.predict_one(tfidf.transform_one(sentence)) == label)
    if (LR.predict_one(tfidf.transform_one(sentence)) != label):
        sum+=1
        print(sum)

    tfidf = tfidf.learn_one(sentence)
    #print(tfidf.transform_one(sentence))
    LR = LR.learn_one(tfidf.transform_one(sentence),label)
    



    #print(label)
    #print(msg.offset)
    if(msg.offset == lastOffset-1):
        print(lastOffset)
        accuracy = (1 - sum/lastOffset) * 100
        print("accuracy = {}% ".format(accuracy))
        break


