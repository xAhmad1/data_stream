# This code will filter the tweets inside the raw-tweets topic into two other topics : en-tweets and fr-tweets
from kafka import KafkaConsumer, KafkaProducer, TopicPartition

# Import sys module
import sys
size =100
# Define server with port
bootstrap_servers = ['localhost:9092']

# Define topic name from where the message will recieve
topicName = 'raw-tweets'
tp = TopicPartition(topicName,0)

producer = KafkaProducer(bootstrap_servers='localhost:9092')
consumer = KafkaConsumer ( group_id=None, auto_offset_reset='earliest', bootstrap_servers =bootstrap_servers)
consumer.assign([tp])
consumer.seek_to_end(tp)
lastOffset = consumer.position(tp)
consumer.seek_to_beginning(tp)

# Read and print message from consumer
s=0
nb_eng = 0
nb_fr = 0
for msg in consumer:
    # s+=1
    lang = (msg.value[1:3]).decode()
    if (lang == "en"):
        producer.send('en-tweets', msg.value[4:])
        nb_eng += 1
    elif (lang == "fr"):
        producer.send('fr-tweets', msg.value[4:])
        nb_fr += 1
    if(msg.offset == lastOffset-1):
        print(nb_eng)
        print(nb_fr)
        break
    