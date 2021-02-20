#!/usr/bin/env python
# coding: utf-8

# # Region Imports

# In[184]:


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

# In[207]:


def basic_cleaning(text):
    no_punctuation = re.sub(r'[^\w\s]','',text)
    no_punctuation = re.sub('_',' ',no_punctuation)
    no_linebreakers = re.sub("[\n\r]", " ", no_punctuation)  
    no_stopwords = " ".join([i for i in no_linebreakers.split() if i.lower() not in set(stopwords.words('english'))])
    no_numbers = re.sub(r'\d+', '', no_stopwords)
    return no_numbers

def basic_cleaning_ign_underscore(text):
    no_punctuation = re.sub(r'[^\w\s]','',text)
    no_linebreakers = re.sub("[\n\r]", " ", no_punctuation)  
    no_stopwords = " ".join([i for i in no_linebreakers.split() if i.lower() not in set(stopwords.words('english'))])
    no_numbers = re.sub(r'\d+', '', no_stopwords)
    return no_numbers

def pos(text):
    return " ".join([word for word, tag in nltk.pos_tag(text.split()) if tag in ['NN','NNP','NNPS','NNS']])

def stem(text):
    stemmer = PorterStemmer()
    stem = " ".join([stemmer.stem(i) for i in text.split()])
    return stem

def lem(text):
    wordnet_lemmatizer = WordNetLemmatizer()
    lem = " ".join([wordnet_lemmatizer.lemmatize(i) for i in text.split()])
    return lem

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
    
    
def models(train, train_doc_term_matrix, test_doc_term_matrix, dictionary):
    lsa = gensim.models.lsimodel.LsiModel
    lsamodel = lsa(train_doc_term_matrix, num_topics=8, id2word = dictionary)
    lsatopics = lsamodel.show_topics(num_words=20,formatted=False)
    print("LSA-Topics:")
    pprint(lsatopics)
    
    # Creating the object for LDA model using gensim library
    lda = gensim.models.ldamodel.LdaModel
    # Running and Trainign LDA model on the document term matrix.
    ldamodel = lda(train_doc_term_matrix, num_topics=8, id2word = dictionary, passes = 20, iterations = 100)
    ldatopics = ldamodel.show_topics(num_words=20,formatted=False)
    print("LDA-Topics:")
    pprint(ldatopics)
    
    hdp = gensim.models.hdpmodel.HdpModel
    hdpmodel = hdp(train_doc_term_matrix, id2word = dictionary)
    hdptopics = hdpmodel.show_topics(num_topics=8 ,num_words=20,formatted=False)
    print("HDP-Topics:")
    pprint(hdptopics)
    
    lsatopics = [[word for word, prob in topic] for topicid, topic in lsatopics]
    ldatopics = [[word for word, prob in topic] for topicid, topic in ldatopics]
    hdptopics = [[word for word, prob in topic] for topicid, topic in hdptopics]
    
    
    lsa_coherence = CoherenceModel(topics=lsatopics, texts=train, dictionary=dictionary, window_size=10).get_coherence()
    print("LSA-Coherence",lsa_coherence)
    lda_coherence = CoherenceModel(topics=ldatopics, texts=train, dictionary=dictionary, window_size=10).get_coherence()
    print("LDA-Coherence",lda_coherence)
    hdp_coherence = CoherenceModel(topics=hdptopics, texts=train, dictionary=dictionary, window_size=10).get_coherence()
    print("HDP-Coherence",hdp_coherence)
    
    evaluate_bar_graph([lsa_coherence, lda_coherence, hdp_coherence], ['LSA', 'LDA', 'HDP'])
    
def models_infer(train, train_doc_term_matrix, test, dictionary):
    lsa = gensim.models.lsimodel.LsiModel
    lsamodel = lsa(train_doc_term_matrix, num_topics=8, id2word = dictionary)
    lsatopics = lsamodel.show_topics(num_words=20,formatted=False)
    print("LSA-Topics:")
    pprint(lsatopics)
    
    # Creating the object for LDA model using gensim library
    lda = gensim.models.ldamodel.LdaModel
    # Running and Trainign LDA model on the document term matrix.
    ldamodel = lda(train_doc_term_matrix, num_topics=8, id2word = dictionary, passes = 20, iterations = 100)
    ldatopics = ldamodel.show_topics(num_words=20,formatted=False)
    print("LDA-Topics:")
    pprint(ldatopics)
    
    hdp = gensim.models.hdpmodel.HdpModel
    hdpmodel = hdp(train_doc_term_matrix, id2word = dictionary)
    hdptopics = hdpmodel.show_topics(num_topics=8 ,num_words=20,formatted=False)
    print("HDP-Topics:")
    pprint(hdptopics)
    
    lsatopics = [[word for word, prob in topic] for topicid, topic in lsatopics]
    ldatopics = [[word for word, prob in topic] for topicid, topic in ldatopics]
    hdptopics = [[word for word, prob in topic] for topicid, topic in hdptopics]
    
    
    lsa_coherence = CoherenceModel(topics=lsatopics, texts=train, dictionary=dictionary, window_size=10).get_coherence()
    print("LSA-Coherence",lsa_coherence)
    lda_coherence = CoherenceModel(topics=ldatopics, texts=train, dictionary=dictionary, window_size=10).get_coherence()
    print("LDA-Coherence",lda_coherence)
    hdp_coherence = CoherenceModel(topics=hdptopics, texts=train, dictionary=dictionary, window_size=10).get_coherence()
    print("HDP-Coherence",hdp_coherence)
    
    evaluate_bar_graph([lsa_coherence, lda_coherence, hdp_coherence], ['LSA', 'LDA', 'HDP'])
    #INFER
    print("\nLSA INFER:\n")
    for doc in test:
        bow = dictionary.doc2bow(doc)
        print(lsamodel[bow])
        print("\nRecoomended:",sorted(lsamodel[bow], key=lambda x: abs(x[1]), reverse=True)[0])
    print("\nLDA INFER:\n")
    for doc in test:
        bow = dictionary.doc2bow(doc)
        print(ldamodel[bow])
        print("\nRecoomended:",sorted(ldamodel[bow], key=lambda x: abs(x[1]), reverse=True)[0])
    print("\nHDP INFER:\n")
    for doc in test:
        bow = dictionary.doc2bow(doc)
        print(hdpmodel[bow])
        print("\nRecoomended:",sorted(hdpmodel[bow], key=lambda x: abs(x[1]), reverse=True)[0])
    return lsamodel, ldamodel, hdpmodel


# # Loading Dataset

# In[186]:


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


# # T4

# # POS

# In[187]:


#normalização base, sem pontuação, stopwords e linebreakers
docs_normalized = [basic_cleaning(pos(doc['text'])).split() for doc in reviews_categorized]

train = docs_normalized[10:]
#10 primeiros para treino
test = docs_normalized[:10]

#construção do dicionario com os dados de treino
dictionary = corpora.Dictionary(train)
#matrizes documentos-termo
train_doc_term_matrix = [dictionary.doc2bow(doc) for doc in train]
test_doc_term_matrix = [dictionary.doc2bow(doc) for doc in test]

models(train, train_doc_term_matrix, test_doc_term_matrix, dictionary)


# # Stemming

# In[188]:


#normalização base, sem pontuação, stopwords e linebreakers
docs_normalized = [basic_cleaning(stem(pos(doc['text']))).split() for doc in reviews_categorized]

train = docs_normalized[10:]
#10 primeiros para treino
test = docs_normalized[:10]

#construção do dicionario com os dados de treino
dictionary = corpora.Dictionary(train)
#matrizes documentos-termo
train_doc_term_matrix = [dictionary.doc2bow(doc) for doc in train]
test_doc_term_matrix = [dictionary.doc2bow(doc) for doc in test]

models(train, train_doc_term_matrix, test_doc_term_matrix, dictionary)


# # Lemmatize

# In[189]:


#normalização base, sem pontuação, stopwords e linebreakers
docs_normalized = [basic_cleaning(lem(pos(doc['text']))).split() for doc in reviews_categorized]

train = docs_normalized[10:]
#10 primeiros para treino
test = docs_normalized[:10]

#construção do dicionario com os dados de treino
dictionary = corpora.Dictionary(train)
#matrizes documentos-termo
train_doc_term_matrix = [dictionary.doc2bow(doc) for doc in train]
test_doc_term_matrix = [dictionary.doc2bow(doc) for doc in test]

models(train, train_doc_term_matrix, test_doc_term_matrix, dictionary)


# # Chunking

# Primeira abordagem

# In[190]:


import spacy
nlp = spacy.load('en_core_web_sm')

reviews_entities = []

for review in reviews_categorized:
    doc = nlp(review["text"])
    chunk = [basic_cleaning(entity.text) for entity in doc.noun_chunks]
    chunk = [c for c in chunk if c.strip()]
    reviews_entities.append(chunk)
print(reviews_categorized[0]["text"])
print("-"*50)
print(reviews_entities[0])


# In[191]:


train = reviews_entities[10:]
#10 primeiros para treino
test = reviews_entities[:10]

#construção do dicionario com os dados de treino
dictionary = corpora.Dictionary(train)
#matrizes documentos-termo
train_doc_term_matrix = [dictionary.doc2bow(doc) for doc in train]
test_doc_term_matrix = [dictionary.doc2bow(doc) for doc in test]

models(train, train_doc_term_matrix, test_doc_term_matrix, dictionary)


# Segunda abordagem

# In[192]:


import spacy
nlp = spacy.load('en_core_web_sm')

reviews_entities = []

for review in reviews_categorized:
    doc = nlp(review["text"])
    chunk = [basic_cleaning(entity.text) for entity in doc.noun_chunks]
    chunk = [c for c in chunk if c.strip()]
    #reviews_entities.append(chunk)
    text = review["text"]
    for c in chunk:
        text = re.sub(c,re.sub(' ','_',c),text) 
    reviews_entities.append(basic_cleaning_ign_underscore(text).split())
print(reviews_categorized[0]["text"])
print("-"*50)
print(reviews_entities[0])


# In[193]:


train = reviews_entities[10:]
#10 primeiros para treino
test = reviews_entities[:10]

#construção do dicionario com os dados de treino
dictionary = corpora.Dictionary(train)
#matrizes documentos-termo
train_doc_term_matrix = [dictionary.doc2bow(doc) for doc in train]
test_doc_term_matrix = [dictionary.doc2bow(doc) for doc in test]

models(train, train_doc_term_matrix, test_doc_term_matrix, dictionary)


# # TF-IDF

# -> POS

# In[194]:


#normalização base, sem pontuação, stopwords e linebreakers
docs_normalized = [basic_cleaning(pos(doc['text'])).split() for doc in reviews_categorized]

train = docs_normalized[10:]
#10 primeiros para treino
test = docs_normalized[:10]

dictionary = corpora.Dictionary(train)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in train]

tfidf = gensim.models.tfidfmodel.TfidfModel
tfidfmodel = tfidf(doc_term_matrix, id2word = dictionary)
#print(tfidfmodel.id2word)
#print(tfidfmodel.dfs)

voc = {}
for i in range(len(tfidfmodel.id2word)):
    voc[tfidfmodel.id2word[i]] = tfidfmodel.idfs[i]
    
sel_features=sorted(voc, key=voc.__getitem__, reverse=True)[:6000]

voc2 = {}
for i in range(len(tfidfmodel.id2word)):
    voc2[tfidfmodel.id2word[i]] = tfidfmodel.dfs[i]
    
def select(doc, voc):
    selected = [i for i in doc if i in voc and i in voc2 and voc2[i] <= 195]
    return selected

#Applying models

doc_clean = [select(doc, sel_features) for doc in docs_normalized]
train = doc_clean[10:]
#10 primeiros para treino
test = doc_clean[:10]
#construção do dicionario com os dados de treino
dictionary = corpora.Dictionary(doc_clean)
#matrizes documentos-termo
train_doc_term_matrix = [dictionary.doc2bow(doc) for doc in train]
test_doc_term_matrix = [dictionary.doc2bow(doc) for doc in test]

models(train, train_doc_term_matrix, test_doc_term_matrix, dictionary)


# -> POS + Stem

# In[196]:


#normalização base, sem pontuação, stopwords e linebreakers
docs_normalized = [basic_cleaning(stem(pos(doc['text']))).split() for doc in reviews_categorized]

train = docs_normalized[10:]
#10 primeiros para treino
test = docs_normalized[:10]

dictionary = corpora.Dictionary(train)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in train]

tfidf = gensim.models.tfidfmodel.TfidfModel
tfidfmodel = tfidf(doc_term_matrix, id2word = dictionary)
#print(tfidfmodel.id2word)
#print(tfidfmodel.dfs)

voc = {}
for i in range(len(tfidfmodel.id2word)):
    voc[tfidfmodel.id2word[i]] = tfidfmodel.idfs[i]
    
sel_features=sorted(voc, key=voc.__getitem__, reverse=True)[:6000]

voc2 = {}
for i in range(len(tfidfmodel.id2word)):
    voc2[tfidfmodel.id2word[i]] = tfidfmodel.dfs[i]
    
def select(doc, voc):
    selected = [i for i in doc if i in voc and i in voc2 and voc2[i] <= 195]
    return selected

#Applying models

doc_clean = [select(doc, sel_features) for doc in docs_normalized]
train = doc_clean[10:]
#10 primeiros para treino
test = doc_clean[:10]
#construção do dicionario com os dados de treino
dictionary = corpora.Dictionary(doc_clean)
#matrizes documentos-termo
train_doc_term_matrix = [dictionary.doc2bow(doc) for doc in train]
test_doc_term_matrix = [dictionary.doc2bow(doc) for doc in test]

models(train, train_doc_term_matrix, test_doc_term_matrix, dictionary)


# -> POS + Lemmas

# In[195]:


#normalização base, sem pontuação, stopwords e linebreakers
docs_normalized = [basic_cleaning(lem(pos(doc['text']))).split() for doc in reviews_categorized]

train = docs_normalized[10:]
#10 primeiros para treino
test = docs_normalized[:10]

dictionary = corpora.Dictionary(train)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in train]

tfidf = gensim.models.tfidfmodel.TfidfModel
tfidfmodel = tfidf(doc_term_matrix, id2word = dictionary)
#print(tfidfmodel.id2word)
#print(tfidfmodel.dfs)

voc = {}
for i in range(len(tfidfmodel.id2word)):
    voc[tfidfmodel.id2word[i]] = tfidfmodel.idfs[i]
    
sel_features=sorted(voc, key=voc.__getitem__, reverse=True)[:6000]

voc2 = {}
for i in range(len(tfidfmodel.id2word)):
    voc2[tfidfmodel.id2word[i]] = tfidfmodel.dfs[i]
    
def select(doc, voc):
    selected = [i for i in doc if i in voc and i in voc2 and voc2[i] <= 195]
    return selected

#Applying models

doc_clean = [select(doc, sel_features) for doc in docs_normalized]
train = doc_clean[10:]
#10 primeiros para treino
test = doc_clean[:10]
#construção do dicionario com os dados de treino
dictionary = corpora.Dictionary(doc_clean)
#matrizes documentos-termo
train_doc_term_matrix = [dictionary.doc2bow(doc) for doc in train]
test_doc_term_matrix = [dictionary.doc2bow(doc) for doc in test]

models(train, train_doc_term_matrix, test_doc_term_matrix, dictionary)


# -> Chunking

# In[197]:


import spacy
nlp = spacy.load('en_core_web_sm')

reviews_entities = []

for review in reviews_categorized:
    doc = nlp(review["text"])
    chunk = [basic_cleaning(entity.text) for entity in doc.noun_chunks]
    chunk = [c for c in chunk if c.strip()]
    reviews_entities.append(chunk)

#normalização base, sem pontuação, stopwords e linebreakers
#docs_normalized = [basic_cleaning(pos(doc['text'])).split() for doc in reviews_categorized]

train = reviews_entities[10:]
#10 primeiros para treino
test = reviews_entities[:10]

dictionary = corpora.Dictionary(train)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in train]

tfidf = gensim.models.tfidfmodel.TfidfModel
tfidfmodel = tfidf(doc_term_matrix, id2word = dictionary)
#print(tfidfmodel.id2word)
#print(tfidfmodel.dfs)

voc = {}
for i in range(len(tfidfmodel.id2word)):
    voc[tfidfmodel.id2word[i]] = tfidfmodel.idfs[i]
    
sel_features=sorted(voc, key=voc.__getitem__, reverse=True)[:6000]

voc2 = {}
for i in range(len(tfidfmodel.id2word)):
    voc2[tfidfmodel.id2word[i]] = tfidfmodel.dfs[i]
    
def select(doc, voc):
    selected = [i for i in doc if i in voc and i in voc2 and voc2[i] <= 195]
    return selected

#Applying models

doc_clean = [select(doc, sel_features) for doc in reviews_entities]
train = doc_clean[10:]
#10 primeiros para treino
test = doc_clean[:10]
#construção do dicionario com os dados de treino
dictionary = corpora.Dictionary(doc_clean)
#matrizes documentos-termo
train_doc_term_matrix = [dictionary.doc2bow(doc) for doc in train]
test_doc_term_matrix = [dictionary.doc2bow(doc) for doc in test]

models(train, train_doc_term_matrix, test_doc_term_matrix, dictionary)


# # T5

# In[208]:


import spacy
nlp = spacy.load('en_core_web_sm')

reviews_entities = []

for review in reviews_categorized:
    doc = nlp(review["text"])
    chunk = [basic_cleaning(entity.text) for entity in doc.noun_chunks]
    chunk = [c for c in chunk if c.strip()]
    reviews_entities.append(chunk)

#normalização base, sem pontuação, stopwords e linebreakers
#docs_normalized = [basic_cleaning(pos(doc['text'])).split() for doc in reviews_categorized]

train = reviews_entities[10:]
#10 primeiros para treino
test = reviews_entities[:10]

dictionary = corpora.Dictionary(train)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in train]

tfidf = gensim.models.tfidfmodel.TfidfModel
tfidfmodel = tfidf(doc_term_matrix, id2word = dictionary)
#print(tfidfmodel.id2word)
#print(tfidfmodel.dfs)

voc = {}
for i in range(len(tfidfmodel.id2word)):
    voc[tfidfmodel.id2word[i]] = tfidfmodel.idfs[i]
    
sel_features=sorted(voc, key=voc.__getitem__, reverse=True)[:6000]

voc2 = {}
for i in range(len(tfidfmodel.id2word)):
    voc2[tfidfmodel.id2word[i]] = tfidfmodel.dfs[i]
    
def select(doc, voc):
    selected = [i for i in doc if i in voc and i in voc2 and voc2[i] <= 195]
    return selected

#Applying models

doc_clean = [select(doc, sel_features) for doc in reviews_entities]
train = doc_clean[10:]
#10 primeiros para treino
test = doc_clean[:10]
#construção do dicionario com os dados de treino
dictionary = corpora.Dictionary(doc_clean)
#matrizes documentos-termo
train_doc_term_matrix = [dictionary.doc2bow(doc) for doc in train]
test_doc_term_matrix = [dictionary.doc2bow(doc) for doc in test]

lsamodel, ldamodel, hdpmodel = models_infer(train, train_doc_term_matrix, test, dictionary)


# In[209]:


hdpmodel.show_topics(num_topics=-1)


# In[199]:


for i, rev in enumerate(reviews_categorized[:10]):
    print("Review Number:",i, "Category:", rev["topic"])


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
