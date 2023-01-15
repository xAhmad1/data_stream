

from kafka import KafkaConsumer, KafkaProducer, TopicPartition
from river import linear_model
from river import feature_extraction
from river import optim
import re
import unicodedata
from nltk.corpus import stopwords
tfidf = feature_extraction.TFIDF()


consumer = KafkaConsumer(bootstrap_servers='localhost:9092')
producer = KafkaProducer(bootstrap_servers='localhost:9092')

topicName = "tweets-labeled-demo"
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


for msg in consumer:
    sentence = (msg.value[4:]).decode()
    sentence = clean_text(sentence)
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


