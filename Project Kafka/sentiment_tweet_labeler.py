# This code is used ot label tweets using a pretrained transformer model

# Imports
from kafka import KafkaConsumer, KafkaProducer, TopicPartition
from transformers import pipeline

# Defining the model pipeline
sent_pipe = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis" , max_length = 128)

# Defining the consumer and producer
consumer = KafkaConsumer(bootstrap_servers='localhost:9092')
producer = KafkaProducer(bootstrap_servers='localhost:9092')

# Defining Topics to consume from
topics = ["en-tweets-demo", "fr-tweets-demo"]

# Read and print message from consumer
for topicName in topics:
    tp = TopicPartition(topicName,0)
    consumer.assign([tp])

    #Getting the size of the topic
    consumer.seek_to_end(tp)
    lastOffset = consumer.position(tp)
    consumer.seek_to_beginning(tp)

    # Predicting the sentiments of the tweet and appending it to a tweets-labeled-demo topic
    for msg in consumer:
        sentiment = sent_pipe(msg.value.decode())[0]['label']
        producer.send('tweets-labeled-demo',str.encode(sentiment) + msg.value + str.encode("\n"))
        if(msg.offset == lastOffset-1):
            break