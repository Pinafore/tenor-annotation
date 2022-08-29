import pickle
import os

# topic model imports
import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim topic model
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy
import nltk

from nltk.corpus import stopwords

# preprocessing
try:
    nltk.data.find('tokenizers/punkt')
    stop_words = stopwords.words('english')

except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# from nltk import word_tokenize, sent_tokenize


from nltk.stem import PorterStemmer
stemmer = PorterStemmer()


class TopicModel():
    
    def __init__(self):
        
        self.documents = None
        self.corpus = None
        self.id2word = None
        self.lda_model = None
        # self.model_dir_path = None

        # self.stop_words = stopwords.words('english')
        self.mallet_path = '/fs/clip-quiz/amao/Github/alto-boot/mallet-2.0.8'

    
    def save_model(self, save_path):
        pickle.dump(self.lda_model, open(save_path, "wb"))
    
    def load_model(self, model_dir_path):
        # self.model_dir_path = model_dir_path
        
        self.corpus = pickle.load(open(os.path.join(model_dir_path, 'corpus.pkl'), "rb"))
#         self.documents = pickle.load(open(os.path.join(model_dir_path, 'corpus.pkl'), "rb"))
        self.id2word = pickle.load(open(os.path.join(model_dir_path, 'id2word.pkl'), "rb"))
        self.lda_model = pickle.load(open(os.path.join(model_dir_path, 'lda_model.pkl'), "rb"))

    # make dictionary, corpus 
    def preprocess(self, documents, model_dir_path):
        
        def sent_to_words(sentences):
            for sentence in sentences:
                yield(gensim.utils.simple_preprocess(str(sentence).encode('utf-8'), deacc=True))  # deacc=True removes punctuations

        # Define functions for stopwords, bigrams, trigrams and lemmatization
        def remove_stopwords(texts):
            return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

        def make_bigrams(texts):
            return [bigram_mod[doc] for doc in texts]

        def make_trigrams(texts):
            return [trigram_mod[bigram_mod[doc]] for doc in texts]

        def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
            """https://spacy.io/api/annotation"""
            texts_out = []
            for sent in texts:
                doc = nlp(" ".join(sent)) 
                texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
            return texts_out

        print('creating features...')
        data = documents
        
        data_words = list(sent_to_words(data))

        # Build the bigram and trigram models
        bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
        trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

        # Faster way to get a sentence clubbed as a trigram/bigram
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)

        # Remove Stop Words
        data_words_nostops = remove_stopwords(data_words)

        # Form Bigrams
        data_words_bigrams = make_bigrams(data_words_nostops)

        # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

        # Do lemmatization keeping only noun, adj, vb, adv
        data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
        
        print('creating dictionary and corpus...')
        # Create Dictionary
        id2word = corpora.Dictionary(data_lemmatized)
        self.id2word = id2word

        # Create Corpus
        texts = data_lemmatized

        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]
        self.corpus = corpus

        # save 
        # self.model_dir_path = model_dir_path
        if not os.path.exists(model_dir_path):
            os.mkdir(model_dir_path)
        pickle.dump(corpus, open(os.path.join(model_dir_path, 'corpus.pkl'), "wb"))
        # pickle.dump(documents, open(os.path.join(model_dir_path, 'documents.pkl'), "wb"))
        pickle.dump(id2word, open(os.path.join(model_dir_path, 'id2word.pkl'), "wb"))

        print('finished preprocessing!')
    
    def save(self, model_dir_path):
        if not os.path.exists(model_dir_path):
            os.mkdir(model_dir_path)
        pickle.dump(self.corpus, open(os.path.join(model_dir_path, 'corpus.pkl'), "wb"))
        # pickle.dump(documents, open(os.path.join(model_dir_path, 'documents.pkl'), "wb"))
        pickle.dump(self.id2word, open(os.path.join(model_dir_path, 'id2word.pkl'), "wb"))

        pickle.dump(self.lda_model, open(os.path.join(model_dir_path, 'lda_model.pkl'), "wb"))



    def train(self, model_dir_path):
        # mallet_path = 'mallet-2.0.8/bin/mallet' # update this path
        # lda_model = gensim.models.wrappers.LdaMallet(self.mallet_path, corpus=self.corpus, num_topics=20, id2word=self.id2word)
        lda_model = gensim.models.ldamodel.LdaModel(corpus=self.corpus,
                                           id2word=self.id2word,
                                           num_topics=20, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
        self.lda_model = lda_model
        pickle.dump(lda_model, open(os.path.join(model_dir_path, 'lda_model.pkl'), "wb"))

    def get_dominant_topic(self, document_data):

        # get dominant topic, topic keywords

        ldamodel = self.lda_model
        corpus = self.corpus
        # texts = self.documents['snippet']

        # Init output
        sent_topics_df = pd.DataFrame()

        # Get main topic in each document
        for i, row in enumerate(ldamodel[corpus]):
            row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = ldamodel.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
                else:
                    break
        # sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
        sent_topics_df.columns = ['dominant_topic', 'dominant_topic_percent', 'topic_keywords']

        # Add original text to the end of the output
        # contents = pd.Series(texts)
        # sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        # return(sent_topics_df)

        # self.documents['dominant_topic']
        sent_topics_df = pd.concat([document_data, sent_topics_df], axis=1)
        return(sent_topics_df)




    def update_model():
        return
    