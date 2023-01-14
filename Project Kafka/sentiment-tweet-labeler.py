from kafka import KafkaConsumer, KafkaProducer, TopicPartition
from transformers import pipeline
import json

sent_pipe = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")
consumer = KafkaConsumer(bootstrap_servers='localhost:9092')
producer = KafkaProducer(bootstrap_servers='localhost:9092')

topics = ["en-tweets", "fr-tweets"]

# Read and print message from consumer
for topicName in topics:
    tp = TopicPartition(topicName,0)
    consumer.assign([tp])

    #Getting the size of the topic
    consumer.seek_to_end(tp)
    lastOffset = consumer.position(tp)
    consumer.seek_to_beginning(tp)

    for msg in consumer:
        sentiment = sent_pipe(msg.value.decode())[0]['label']
        producer.send('tweets-labeled',str.encode(sentiment) + msg.value + str.encode("\n"))
        if(msg.offset == lastOffset-1):
            break