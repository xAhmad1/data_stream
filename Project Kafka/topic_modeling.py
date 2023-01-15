# This code does topic modeling using LDA

# Imports
from kafka import KafkaConsumer, TopicPartition
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import unicodedata
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import re
from wordcloud import WordCloud
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
#import nltk
#nltk.download('wordnet')
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from spacy.lang.en.stop_words import STOP_WORDS as en_stop
stp_list = list(fr_stop) + list(en_stop)
def clean_text(text1):
    rt = text1[0:3]
    if (rt == "RT "):
        text1 = re.sub('RT @.*:','',text1)
    text1 = text1.lower()
    text1 = unicodedata.normalize('NFKD',text1).encode('ascii', errors='ignore').decode('utf-8')
    text1 =  re.sub('[^A-Za-z]',' ',text1)
    text1 = re.sub(r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', ' ',text1)
    text1 = re.sub('https', ' ',text1)
    text1 = re.sub(r'\b\w{1,3}\b', '', text1)
    text1 = re.sub(r'\s+', ' ',text1)
    text1 = ' '.join([word for word in text1.split() if word not in stp_list])
    return text1
df_tweets['tweet'] = df_tweets.iloc[:,0].apply(lambda x: clean_text(x))

# Transform the data using tfidf
X_train, X_test, y_train, y_test = train_test_split(df_tweets['tweet'], df_tweets['label'], test_size=0.33, random_state=42)
cv1 = TfidfVectorizer(lowercase=True, min_df=3,  max_features=None,
                      strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                      ngram_range=(1, 3))
vectorized1 = cv1.fit_transform(X_train)
vectorized2 = cv1.transform(X_test)

# Make a list of words used in the tweets
list1  = df_tweets.iloc[:,0].apply(lambda x: x.split())
wordS_list = [item for list2 in list1 for item in list2]

# Make a dictionary of words used in the tweets with their counts
words_counts = {}
for word in wordS_list:
    if (word in words_counts.keys()):
        words_counts[word] +=1
    else:
        words_counts[word] = 1


# Sort the dictionary
sorted_words_counts = sorted(words_counts.items(), key=lambda x:x[1] , reverse= True)
top_50_dict = dict(sorted_words_counts[:50])

# Create a word cloud from the dictionary
wordcloud = WordCloud(colormap = 'Accent', background_color = 'black')\
.generate_from_frequencies(top_50_dict)

#plot with matplotlib
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig('top_50_dict.png')

plt.show()

# Attempting to fit a model using LDA : Latent Dirchlet Allocation

from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary

## Create a bag of words from the dictionary
text_dict = Dictionary(list1)
tweets_bow = [text_dict.doc2bow(tweet) for tweet in list1]

## Fitting an LDA model

k = 5
tweets_lda = LdaModel(tweets_bow,
                      num_topics = k,
                      id2word = text_dict,
                      random_state = 1,
                      passes=50)

# visualising the topics
import pyLDAvis.gensim
visualisation = pyLDAvis.gensim.prepare(tweets_lda, tweets_bow, dictionary=tweets_lda.id2word)
pyLDAvis.save_html(visualisation, 'LDA_Visualization.html')