#!/usr/bin/env python
# coding: utf-8

# # Region Imports

# In[97]:


import json
import random
import string
import re
import gensim
import operator
import nltk
import matplotlib.pyplot as plt
import numpy as np
from gensim import corpora
from nltk.corpus import stopwords
from pprint import pprint
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from gensim.models import CoherenceModel


# # Region Functions

# In[98]:


def basic_cleaning(text):
    no_punctuation = re.sub(r'[^\w\s]','',text)
    no_punctuation = re.sub('_',' ',no_punctuation)
    no_linebreakers = re.sub("[\n\r]", " ", no_punctuation)  
    no_stopwords = " ".join([i for i in no_linebreakers.split() if i.lower() not in set(stopwords.words('english'))])
    return no_stopwords

def evaluate_bar_graph(coherences, indices):
    """
    Function to plot bar graph.
    coherences: list of coherence values
    indices: Indices to be used to mark bars. Length of this and coherences should be equal.
    """
    assert len(coherences) == len(indices)
    n = len(coherences)
    x = np.arange(n)
    plt.bar(x, coherences, width=0.2, tick_label=indices, align='center')
    plt.xlabel('Models')
    plt.ylabel('Coherence Value')


# # Loading Dataset

# Categorização dos documentos, é criado um novo dicionário das riviews com a categoria corresponde.

# In[99]:


reviews_categorized = []
topics = ['Books','Cars','Computers','Cookware','Hotels','Movies','Music','Phones']
for i, doc in enumerate(open('../TM/data/en/SFU_Review_Corpus.json', encoding="utf-8")):
    review = {}
    review['text'] = json.loads(doc)['text']
    if(i<50):
        review['topic'] = topics[0]
    elif(i>=50 and i<100):
        review['topic'] = topics[1]
    elif(i>=100 and i<150):
        review['topic'] = topics[2]
    elif(i>=150 and i<200):
        review['topic'] = topics[3]
    elif(i>=200 and i<250):
        review['topic'] = topics[4]
    elif(i>=250 and i<300):
        review['topic'] = topics[5]
    elif(i>=300 and i<350):
        review['topic'] = topics[6]
    elif(i>=350 and i<400):
        review['topic'] = topics[7]
    
    reviews_categorized.append(review)
#Documentos vem em blocos de categorias, para evitar isso baralhamos os documentos
random.seed(18)
random.shuffle(reviews_categorized)


# In[100]:


topic_normalized = [doc['topic'] for doc in reviews_categorized]
train = topic_normalized[10:]
#10 primeiros para treino
test = topic_normalized[:10]
print("Categorias dos documentos guardamos para inferencia:", test)


# # T2

# In[101]:


#normalização base, sem pontuação, stopwords e linebreakers
docs_normalized = [basic_cleaning(doc['text']).split() for doc in reviews_categorized]

train = docs_normalized[10:]
#10 primeiros para treino
test = docs_normalized[:10]


# In[102]:


#construção do dicionario com os dados de treino
dictionary = corpora.Dictionary(train)
#matrizes documentos-termo
train_doc_term_matrix = [dictionary.doc2bow(doc) for doc in train]
test_doc_term_matrix = [dictionary.doc2bow(doc) for doc in test]


# # T3

# 1.LSA (nº de tópicos = 8 , nº de palavras por tópico =20)

# In[103]:


lsa = gensim.models.lsimodel.LsiModel
lsamodel = lsa(train_doc_term_matrix, num_topics=8, id2word = dictionary)


# In[104]:


lsatopics = lsamodel.show_topics(num_words=20,formatted=False)


# In[105]:


for topicid, topic in lsatopics:
    t = []
    for w,p in topic:
        t.append(w)
    print(topicid, ": ",t)


# 2.LDA (nº de tópicos = 8, nº de palavras por tópico =20, nº de passagens = 20, nº iterações = 100))

# In[106]:


# Creating the object for LDA model using gensim library
lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel = lda(train_doc_term_matrix, num_topics=8, id2word = dictionary, passes=20, iterations=100)


# In[107]:


ldatopics = ldamodel.show_topics(num_words=20,formatted=False)


# In[108]:


for topicid, topic in ldatopics:
    t = []
    for w,p in topic:
        t.append(w)
    print(topicid, ": ",t)


# In[109]:


import pyLDAvis.gensim
pyLDAvis.enable_notebook()
pyLDAvis.gensim.prepare(ldamodel, train_doc_term_matrix, dictionary)


# Evaluation

# In[110]:


lsatopics = [[word for word, prob in topic] for topicid, topic in lsatopics]
ldatopics = [[word for word, prob in topic] for topicid, topic in ldatopics]


# In[111]:


lsa_coherence = CoherenceModel(topics=lsatopics, texts=train, dictionary=dictionary, window_size=10).get_coherence()
print("Coerencias LSA :", lsa_coherence)
lda_coherence = CoherenceModel(topics=ldatopics, texts=train, dictionary=dictionary, window_size=10).get_coherence()
print("Coerencias LDA :",lda_coherence)


# In[112]:


evaluate_bar_graph([lsa_coherence, lda_coherence], ['LSA', 'LDA'])


# 3.WordCloud

# LSA Cloud

# In[113]:


lsa_cloud = {}
for i in lsamodel.show_topics(num_words=20,formatted=False):
    for word in i[1]:
        if word[0] in lsa_cloud:
            lsa_cloud[word[0]] = max(lsa_cloud[word[0]], abs(word[1]))
        else:
            lsa_cloud[word[0]] = abs(word[1])
sorted_lsa_cloud = sorted(lsa_cloud.items(), key=operator.itemgetter(1), reverse=True)
pprint(sorted_lsa_cloud)


# In[114]:


wordcloud = WordCloud(background_color="white", max_words=200)
wordcloud.generate_from_frequencies(lsa_cloud)
plt.figure( figsize=(10,5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# LDA Cloud

# In[115]:


lda_cloud = {}
for i in ldamodel.show_topics(num_words=20,formatted=False):
    for word in i[1]:
        if word[0] in lda_cloud:
            lda_cloud[word[0]] = max(lda_cloud[word[0]], abs(word[1]))
        else:
            lda_cloud[word[0]] = abs(word[1])
sorted_lda_cloud = sorted(lda_cloud.items(), key=operator.itemgetter(1), reverse=True)
pprint(sorted_lda_cloud)


# In[116]:


wordcloud = WordCloud(max_words=200, background_color="black")
wordcloud.generate_from_frequencies(lda_cloud)
plt.figure( figsize=(10,5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# 4.Correspondencia de Tópicos

# LSA: 
# 
# 
# | NumTópico | Tópicos | Categoria |
# | --- | --- | --- |
# | 0 |[ like ,  one ,  car ,  would ,  also ,  get ,  good ,  much ,  time ,  iMac ,  even ,  really ,  dont ,  well ,  new ,  Dell ,  use ,  back ,  want ,  two ]|Cars, Computers | 
# | 1 |[ car ,  iMac ,  Mac ,  Dell ,  PC ,  Apple ,  computer ,  photos ,  iPhoto ,  applications ,  rear ,  seat ,  page ,  1 ,  engine ,  cars ,  machine ,  seats ,  mouse ,  easy ]|Cars, Computers |
# | 2 |[ car ,  track ,  song ,  beat ,  Ras ,  album ,  lyrics ,  Kass ,  Stars ,  rap ,  chorus ,  Dell ,  system ,  spits ,  one ,  us ,  hip ,  5 ,  hop ,  production ]|Music | 
# | 3 |[ Dell ,  1 ,  system ,  iMac ,  Mac ,  Customer ,  2 ,  may ,  Care ,  software ,  System ,  Apple ,  photos ,  drive ,  iPhoto ,  problems ,  performance ,  computer ,  Ras ,  applications ]| Computers, Phones  |
# | 4 |[ car ,  Ras ,  AllClad ,  iMac ,  track ,  Kass ,  Mac ,  room ,  phone ,  one ,  pan ,  Stainless ,  set ,  hotel ,  PC ,  book ,  Apple ,  album ,  dont ,  photos ]| Computers, Cookware, Hotels | 
# | 5 |[ Ras ,  Kass ,  song ,  beat ,  spits ,  Murphy ,  Ice ,  chorus ,  Soul ,  us ,  hop ,  hip ,  lyrics ,  Lee ,  Nelly ,  album ,  room ,  verse ,  metaphors ,  Lil ]| Music |
# | 6 |[ AllClad ,  Stainless ,  pan ,  Steel ,  Fry ,  cookware ,  Pan ,  room ,  pans ,  stainless ,  book ,  set ,  use ,  heat ,  Pans ,  hotel ,  movie ,  film ,  kitchen ,  steel ]| Cookware, Hotels | 
# | 7 |[ phone ,  room ,  handset ,  Panasonic ,  phones ,  battery ,  cordless ,  base ,  hotel ,  handsets ,  Disney ,  system ,  features ,  unit ,  resort ,  ID ,  use ,  caller ,  station ,  pool ]| Hotels, Phones |
# 
# LDA: 
# 
# | NumTópico | Tópicos | Categoria |
# | --- | --- | --- |
# | 0 |['room', 'hotel', 'Disney', 'resort', 'pool', 'stay', 'rooms', 'time', 'two', 'one', 'also', 'Club', 'get', 'like', 'Beach', 'good', 'night', 'area', 'take', 'go']| Hotels | 
# | 1 |['song', 'album', 'one', 'track', 'like', 'beat', 'Dell', 'lyrics', 'good', '1', '5', 'rap', 'songs', 'Ras', 'music', 'even', 'get', 'great', 'time', 'Stars']| Music |
# | 2 |['one', 'computer', 'would', 'like', 'book', 'get', 'use', 'also', 'dont', 'time', 'iMac', 'Apple', 'even', 'pan', 'much', 'AllClad', 'new', 'pans', 'little', 'set']| Computers | 
# | 3 |['phone', 'handset', 'one', 'phones', 'like', 'use', 'Panasonic', 'good', 'base', 'would', 'system', 'battery', 'dont', 'cordless', 'set', '2', 'handsets', 'features', 'time', 'back']| Phones  |
# | 4 |['like', 'one', 'movie', 'get', 'Samurai', 'even', 'book', 'good', 'would', 'go', 'time', 'really', 'well', 'want', 'though', 'see', 'way', 'dont', 'Cruise', 'say']| Movies /Books | 
# | 5 |['movie', 'like', 'film', 'one', 'track', 'album', 'song', 'would', 'beat', 'make', 'kids', 'first', 'Murphy', 'think', 'lyrics', 'way', 'Buddy', 'comes', 'even', 'plot']| Movies |
# | 6 |['like', 'book', 'one', 'Stephanie', 'movie', 'much', 'way', 'dont', 'story', 'never', 'read', 'really', 'know', 'good', 'would', 'time', 'get', 'well', 'could', 'books']|Books | 
# | 7 |['car', 'like', 'one', 'engine', 'also', 'would', 'get', 'good', 'cars', 'rear', 'power', 'even', 'seat', 'much', 'seats', 'drive', 'front', 'driving', 'Ford', '2002']| Cars|
# 
# 

# 5.Variação do número de tópicos

# LSA

# In[117]:


lsa = gensim.models.lsimodel.LsiModel
lsamodel = lsa(train_doc_term_matrix, num_topics=4, id2word = dictionary)
lsa_4_topics = lsamodel.show_topics(num_words=20,formatted=False)


# In[118]:


for topicid, topic in lsa_4_topics:
    t = []
    for w,p in topic:
        t.append(w)
    print(topicid, ": ",t)


# In[119]:


lsa = gensim.models.lsimodel.LsiModel
lsamodel = lsa(train_doc_term_matrix, num_topics=16, id2word = dictionary)
lsa_16_topics = lsamodel.show_topics(num_topics=16,num_words=20, formatted=False)


# In[120]:


for topicid, topic in lsa_16_topics:
    t = []
    for w,p in topic:
        t.append(w)
    print(topicid, ": ",t)


# LDA

# In[121]:


lda = gensim.models.ldamodel.LdaModel
ldamodel = lda(train_doc_term_matrix, num_topics=4, id2word = dictionary, passes=20, iterations=100)
lda_4_topics = ldamodel.show_topics(num_words=20,formatted=False)


# In[122]:


for topicid, topic in lda_4_topics:
    t = []
    for w,p in topic:
        t.append(w)
    print(topicid, ": ",t)


# In[123]:


import pyLDAvis.gensim
pyLDAvis.enable_notebook()
pyLDAvis.gensim.prepare(ldamodel, train_doc_term_matrix, dictionary)


# In[124]:


# Creating the object for LDA model using gensim library
lda = gensim.models.ldamodel.LdaModel
ldamodel = lda(train_doc_term_matrix, num_topics=16, id2word = dictionary, passes=20, iterations=100)
lda_16_topics = ldamodel.show_topics(num_topics=16, num_words=20,formatted=False)


# In[125]:


for topicid, topic in lda_16_topics:
    t = []
    for w,p in topic:
        t.append(w)
    print(topicid, ": ",t)


# In[126]:


import pyLDAvis.gensim
pyLDAvis.enable_notebook()
pyLDAvis.gensim.prepare(ldamodel, train_doc_term_matrix, dictionary)


# LSA 4 Tópicos: 
# 
# | NumTópico | Tópicos | Categoria |
# | --- | --- | --- |
# | 0 |['like', 'one', 'car', 'would', 'also', 'get', 'good', 'much', 'time', 'iMac', 'even', 'really', 'dont', 'well', 'new', 'Dell', 'use', 'back', 'want', 'two']|Cars, Computers | 
# | 1 |['car', 'iMac', 'Mac', 'Dell', 'PC', 'Apple', 'computer', 'photos', 'iPhoto', 'applications', 'rear', 'seat', 'page', '1', 'engine', 'cars', 'machine', 'seats', 'mouse', 'easy']|Cars, Computers |
# | 2 |['car', 'track', 'song', 'beat', 'Ras', 'album', 'lyrics', 'Kass', 'Stars', 'rap', 'chorus', 'Dell', 'system', 'spits', 'one', 'us', 'hip', '5', 'hop', 'production']|Music | 
# | 3 |['Dell', '1', 'system', 'iMac', 'Mac', 'Customer', '2', 'may', 'Care', 'software', 'System', 'Apple', 'photos', 'drive', 'iPhoto', 'problems', 'performance', 'Ras', 'computer', 'applications']| Computers |
# 
# LSA 16 Tópicos: 
# 
# | NumTópico | Tópicos | Categoria |
# | --- | --- | --- |
# | 0 |['like', 'one', 'car', 'would', 'also', 'get', 'good', 'much', 'time', 'iMac', 'even', 'really', 'dont', 'well', 'new', 'Dell', 'use', 'back', 'want', 'two']| Cars | 
# | 1 |['car', 'iMac', 'Mac', 'Dell', 'PC', 'Apple', 'computer', 'photos', 'iPhoto', 'applications', 'rear', 'seat', 'page', '1', 'engine', 'cars', 'machine', 'seats', 'mouse', 'easy']| Computers |
# | 2 |['car', 'track', 'song', 'beat', 'Ras', 'album', 'lyrics', 'Kass', 'Stars', 'rap', 'chorus', 'Dell', 'system', 'spits', 'one', 'us', 'hip', '5', 'hop', 'production']| Music | 
# | 3 |['Dell', '1', 'system', 'iMac', 'Mac', 'Customer', '2', 'may', 'Care', 'software', 'System', 'Apple', 'photos', 'drive', 'iPhoto', 'problems', 'performance', 'Ras', 'computer', 'applications']| Computers  |
# | 4 |['car', 'Ras', 'iMac', 'AllClad', 'track', 'Kass', 'Mac', 'room', 'phone', 'one', 'pan', 'Stainless', 'set', 'hotel', 'PC', 'book', 'Apple', 'album', 'dont', 'photos']| ???? | 
# | 5 |['Ras', 'Kass', 'song', 'beat', 'spits', 'Murphy', 'Ice', 'chorus', 'Soul', 'us', 'hop', 'hip', 'lyrics', 'Lee', 'Nelly', 'album', 'room', 'verse', 'metaphors', 'Lil']| Music |
# | 6 |['AllClad', 'Stainless', 'pan', 'Steel', 'Fry', 'cookware', 'Pan', 'room', 'pans', 'stainless', 'book', 'set', 'use', 'heat', 'Pans', 'hotel', 'movie', 'film', 'kitchen', 'steel']| Cookware | 
# | 7 |['phone', 'room', 'handset', 'Panasonic', 'phones', 'battery', 'base', 'cordless', 'hotel', 'handsets', 'Disney', 'system', 'features', 'unit', 'resort', 'ID', 'caller', 'use', 'station', 'pool']| Phones|
# | 8 |['room', 'film', 'movie', 'phone', 'book', 'hotel', 'story', 'resort', 'Disney', 'pool', 'rooms', 'character', 'also', 'plot', 'two', 'Santa', 'stay', 'Club', 'service', 'like']| Movies/Hotels | 
# | 9 |['computer', 'new', 'iMac', 'phone', 'one', 'Dell', 'Apple', 'drive', 'Mac', 'keyboard', 'iMacs', 'G4', 'two', 'PC', 'OS', 'photos', 'speed', 'easy', 'iPhoto', 'page']| Computers |
# | 10|['Murphy', 'Lee', 'spits', 'chorus', 'Neptunes', 'verse', 'album', 'Ras', 'lyrics', 'sounds', 'St', 'Nelly', 'Stars', 'Pharrell', 'sht', 'really', 'pretty', '1', 'Kass', 'music']| Music | 
# | 11|['Xterra', 'car', 'Taurus', 'book', 'vehicle', 'like', 'engine', 'rear', 'mph', 'time', 'air', 'V6', 'Ford', 'one', 'Dell', 'ride', '1', 'dont', 'much', 'quality']| Cars  |
# | 12|['book', 'film', 'read', 'like', 'Grisham', 'Luke', 'story', 'one', 'novel', 'Taurus', 'cotton', 'Katherine', 'movie', 'Betty', 'family', 'books', 'Stephanie', 'performance', 'films', 'Dell']| Books | 
# | 13|['like', 'Dell', 'Xterra', '1', 'film', 'performance', 'SVT', 'Nissan', 'Focus', 'phone', 'rear', 'system', 'Wow', 'Customer', 'new', 'little', 'may', 'since', 'drive', 'CL']| ???? |
# | 14|['Xterra', 'Taurus', 'Dell', '1', 'SVT', 'Focus', 'car', 'one', 'book', 'V6', 'vehicle', '2002', 'Nissan', 'may', 'film', 'Sable', 'version', 'Customer', 'control', 'new']| Cars | 
# | 15|['Taurus', 'Xterra', 'Ford', 'popcorn', '2002', 'Sable', 'system', 'use', 'book', 'control', 'drive', 'truck', 'AllClad', 'model', 'SUV', 'Whirley', 'corn', 'film', 'Nissan', 'Dell']| ????|
# 
# LDA 4 Tópicos: 
# 
# | NumTópico | Tópicos | Categoria |
# | --- | --- | --- |
# | 0 |['car', 'one', 'like', 'would', 'book', 'also', 'much', 'get', 'time', 'room', 'good', 'even', 'well', 'two', 'dont', 'really', 'back', 'hotel', 'little', 'first']|Hotels| 
# | 1 |['one', 'computer', 'use', 'like', 'get', 'iMac', 'Apple', 'dont', 'would', 'pan', 'AllClad', 'also', 'pans', 'new', 'even', 'set', 'cookware', 'time', 'Mac', 'good']|Computers/Cookware |
# | 2 |['phone', 'one', 'handset', 'like', 'phones', 'good', 'time', 'Panasonic', 'get', 'base', 'use', 'system', 'also', 'Santa', 'cordless', 'handsets', 'Club', 'would', 'really', 'set'] |Phones | 
# | 3 |['movie', 'like', 'album', 'song', 'track', 'one', 'beat', 'lyrics', 'film', 'even', 'good', 'would', 'rap', 'see', 'get', 'great', 'way', 'Im', 'think', 'make']| Musics/Moveis   |
# 
# LDA 16 Tópicos:
# 
# | NumTópico | Tópicos | Categoria |
# | --- | --- | --- |
# | 0 |['one', 'computer', 'would', 'like', 'get', 'Apple', 'dont', 'iMac', 'book', 'really', 'even', 'movie', 'new', 'good', 'much', 'time', 'see', 'system', 'Ive', 'also']| Computers | 
# | 1 |['pan', 'pans', 'AllClad', 'cookware', 'set', 'use', 'one', 'stainless', 'dont', 'like', 'heat', 'Stainless', 'steel', 'cooking', 'pots', 'pot', 'handles', 'even', 'cook', 'dishwasher']| Cookware |
# | 2 |['Dell', 'system', 'memory', 'computer', 'quality', 'also', 'speed', 'Apple', 'USB', 'like', 'drive', 'one', 'several', 'version', 'software', 'XP', 'support', 'great', 'ports', 'get']| Computers | 
# | 3 |['book', 'story', 'movie', 'like', 'books', 'read', 'Buddy', 'characters', 'time', 'think', 'old', 'one', 'character', 'could', 'way', 'much', 'year', 'Tom', 'plot', 'get']| Books/Movies  |
# | 4 |['SVT', 'Focus', 'room', 'hotel', 'would', 'car', 'well', 'one', 'Shyne', 'like', 'Nugget', 'also', 'even', 'two', 'large', 'make', 'time', 'without', 'enough', 'Treasure']| ???? | 
# | 5 |['Benzino', 'like', 'feat', 'Beyond', 'Styles', 'However', 'rapper', 'WI', 'Redemption', 'futuristic', 'two', 'great', 'Calendar', 'eMac', 'Ray', 'Mirren', 'calendar', 'based', 'HipHop', 'Chris']| Musics |
# | 6 |['car', 'like', 'engine', 'seats', 'rear', 'cars', 'power', 'also', 'one', 'Ford', 'seat', 'much', 'front', '2002', 'good', 'interior', 'vehicle', 'driving', 'Taurus', 'drive']|Cars | 
# | 7 |['movie', 'film', 'Santa', 'like', 'Murphy', 'one', 'plot', 'good', 'Christmas', 'character', 'movies', 'would', 'Dr', 'funny', 'Lee', 'see', 'kids', 'bad', 'story', 'really']| Movie|
# | 8 |['Ras', 'Mac', 'iMac', 'Kass', 'like', 'photos', 'iPhoto', 'PC', 'way', 'also', 'many', 'one', 'applications', 'get', 'much', 'easy', 'since', 'Katherine', 'page', 'track']| Computers?? | 
# | 9 |['car', 'like', 'Xterra', 'Nissan', 'book', 'get', 'one', 'Stephanie', 'would', 'good', 'also', 'Neptunes', 'Bow', 'new', 'even', 'Mitsubishi', 'cars', 'Lancer', 'Wow', 'way']| Cars |
# | 10|['Dell', '1', 'system', '2', 'Customer', 'may', 'Care', 'order', 'software', 'standard', 'computer', 'drive', 'address', 'problems', 'performance', 'System', 'want', 'PS2', 'would', 'memory']| Computers | 
# | 11|['phone', 'room', 'one', 'like', 'hotel', 'Disney', 'time', 'get', 'would', 'resort', 'good', 'stay', 'really', 'also', 'rooms', 'handset', 'two', 'pool', 'book', 'dont']| Phones/Hotels  |
# | 12|['room', 'one', 'time', 'like', 'good', 'Samurai', 'hotel', 'much', 'use', 'great', 'even', 'see', 'get', 'large', 'would', 'Cruise', 'really', 'SYM', 'well', 'computer']| ???? | 
# | 13|['like', 'Cat', 'even', 'movie', 'one', 'computer', 'get', 'time', 'would', 'story', 'film', 'way', '5', 'say', 'system', 'book', 'Conrad', 'kids', 'also', 'could']| ???? |
# | 14|['album', 'track', 'song', 'one', 'beat', 'like', 'lyrics', 'rap', 'songs', 'production', 'two', 'first', 'hiphop', 'good', 'music', 'would', 'get', 'sounds', 'tracks', 'Im']| Musics | 
# | 15|['phone', 'one', 'like', 'time', 'good', 'book', 'song', 'would', 'also', 'get', 'Grisham', 'little', 'even', 'Luke', 'really', 'great', 'spits', 'much', 'cotton', 'think']| Phones |
# 

# Evaluation

# In[127]:


lsa_4_topics = [[word for word, prob in topic] for topicid, topic in lsa_4_topics]
lsa_16_topics = [[word for word, prob in topic] for topicid, topic in lsa_16_topics]
lda_4_topics = [[word for word, prob in topic] for topicid, topic in lda_4_topics]
lda_16_topics = [[word for word, prob in topic] for topicid, topic in lda_16_topics]


# In[128]:


print("Coerencia dos tópicos")
lsa_4_coherence = CoherenceModel(topics=lsa_4_topics, texts=train, dictionary=dictionary, window_size=10).get_coherence()
print("LSA- nº de tópicos= 4: ",lsa_4_coherence)
lsa_16_coherence = CoherenceModel(topics=lsa_16_topics, texts=train, dictionary=dictionary, window_size=10).get_coherence()
print("LSA- nº de tópicos= 16: ",lsa_16_coherence)
lda_4_coherence = CoherenceModel(topics=lda_4_topics, texts=train, dictionary=dictionary, window_size=10).get_coherence()
print("LDA- nº de tópicos= 4: ",lda_4_coherence)
lda_16_coherence = CoherenceModel(topics=lda_16_topics, texts=train, dictionary=dictionary, window_size=10).get_coherence()
print("LDA- nº de tópicos= 16: ",lda_16_coherence)


# In[129]:


evaluate_bar_graph([lsa_4_coherence,lsa_coherence,lsa_16_coherence,lda_4_coherence, lda_coherence,lda_16_coherence], 
                   ['LSA-4', 'LSA-8', 'LSA-16', 'LDA-4', 'LDA-8', 'LDA-16'])


# # Links

# https://miningthedetails.com/blog/python/lda/GensimLDA/
# 
# https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
# 
# https://markroxor.github.io/gensim/static/notebooks/gensim_news_classification.html
# 
# https://pdfs.semanticscholar.org/e024/7475102b93143b6de85fff0f3967b209f9f6.pdf
# 
# http://qpleple.com/perplexity-to-evaluate-topic-models/
