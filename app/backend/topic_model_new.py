
# labeled topic model

import tomotopy as tp
from tomotopy.utils import Corpus

import pickle
import os
import re

import numpy as np
import pandas as pd
from pprint import pprint

# spacy for lemmatization
import spacy
import nltk
from nltk.corpus import stopwords

import sklearn
import gensim

# preprocessing
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

stop_words = stopwords.words('english')
# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

class TopicModel():
    '''
    supports preprocessing, save and load, training
    '''
    
    def __init__(self, model_type, min_num_topics, num_iters):
        '''
        valid models: LLDA, SLDA
        '''
        self.documents = None
        self.corpus = None
        self.lda_model = None

        self.model_type = model_type
        if model_type == 'LLDA':
            self.model_lib = tp.LLDAModel
        elif model_type == 'SLDA':  
            self.model_lib = tp.SLDAModel
        else:
            raise Exception("unsupported model!")

        self.label_set = []
        self.num_topics = min_num_topics # minimum number of topics
        self.num_iters = num_iters # number of passes through the corpus


    def save(self, model_dir_path):
        if not os.path.exists(model_dir_path):
            os.mkdir(model_dir_path)
        pickle.dump(self.corpus, open(os.path.join(model_dir_path, 'corpus.pkl'), "wb"))
        model_path = os.path.join(model_dir_path, 'lda_model.bin')
        self.lda_model.save(model_path)

    def load_model(self, model_dir_path):
        self.corpus = pickle.load(open(os.path.join(model_dir_path, 'corpus.pkl'), "rb"))
        model_path = os.path.join(model_dir_path, 'lda_model.bin')
        self.lda_model = self.model_lib.load(model_path)

    def preprocess(self, documents: list, model_dir_path, labels: list=[], fast=True, verbose=True):
        '''
        preprocess text: tokenize, lemmatize, make corpus.

        optionally takes labels
        fast=True: skip making bigrams
        '''
        
        def sent_to_words(sentences):
            # for sentence in sentences:
            #     yield(gensim.utils.simple_preprocess(str(sentence).encode('utf-8'), deacc=True))  # deacc=True removes punctuations
            output = []
            for sentence in sentences:
                tokens = (gensim.utils.simple_preprocess(str(sentence).encode('utf-8'), min_len=1, max_len=30, deacc=True))  # deacc=True removes punctuations
                if len(tokens) == 0:
                    print(sentence, tokens)
                output.append(tokens)

            return output

        # Define functions for stopwords, bigrams, trigrams and lemmatization
        def remove_stopwords(texts):
            return [[word for word in gensim.utils.simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

        def make_bigrams(texts):
            return [bigram_mod[doc] for doc in texts]

        # def make_trigrams(texts):
        #     return [trigram_mod[bigram_mod[doc]] for doc in texts]

        def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
            """https://spacy.io/api/annotation"""
            texts_out = []
            for sent in texts:
                doc = nlp(" ".join(sent)) 
                texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
            return texts_out

        if verbose: 
            print('creating features...')
        data_words = list(sent_to_words(documents))
        data_words = remove_stopwords(data_words)

        if fast == False:
            print('making bigrams...')
            # Build the bigram and trigram models
            bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
            # trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
            # Faster way to get a sentence clubbed as a trigram/bigram
            bigram_mod = gensim.models.phrases.Phraser(bigram)
            # trigram_mod = gensim.models.phrases.Phraser(trigram)
            data_words = make_bigrams(data_words)

            # Do lemmatization keeping only noun, adj, vb, adv
            print('lemmatizing...')
            data_words = lemmatization(data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

        # corpus = Corpus(stopwords=stop_words) # TODO: drop stopwords, tokenize, lemmatize, make ngrams
        corpus = Corpus()

        # make label dict
        label_set = list(set(labels))
        if np.nan in label_set:
            label_set.remove(np.nan)
        if None in label_set:
            label_set.remove(None)
        label_dict = dict()
        for i,label in enumerate(label_set):
            label_dict[label] = i
        self.label_set = label_set


        if self.model_type == 'LLDA':
            for i, ngrams in enumerate(data_words):
                if labels and type(labels[i]) == str:
                    label = labels[i]
                    corpus.add_doc(ngrams, labels=[label])
                else:
                    corpus.add_doc(ngrams)
        elif self.model_type == 'SLDA':
            for i, ngrams in enumerate(data_words):
                y = [0 for _ in range(len(label_set))]
                null_y = [np.nan for _ in range(len(label_set))]
                if labels and type(labels[i]) == str:
                    label = labels[i]
                    y[label_dict[label]] = 1
                    corpus.add_doc(ngrams, y=y)
                    # print(y)
                else:
                    corpus.add_doc(ngrams, y=null_y)
                    # print(null_y)

                # print(corpus[i])
                # if i>10: break
        else:
            raise Exception("unsupported model type!")
        # except Exception as e:
        #     print(e)
        #     print(i, labels[i], type(labels[i]))

        self.corpus = corpus
        if verbose: print('corpus size:', len(corpus))
        assert len(corpus) == len(documents)

        print('model_dir_path', model_dir_path)
        if not os.path.exists(model_dir_path):
            os.mkdir(model_dir_path)
        pickle.dump(self.corpus, open(os.path.join(model_dir_path, 'corpus.pkl'), "wb"))
        
        if verbose: print('finished preprocessing!')
    
    def train(self, model_dir_path, num_topics=None, verbose=True):

        if not num_topics: 

            num_topics = max(self.num_topics, len(self.label_set))
            # if self.label_set:
            # else: num_topics = self.num_topics
        print('num topics:', num_topics)

        if self.model_type == 'LLDA':
            mdl = tp.LLDAModel(k=num_topics)
            # mdl = tp.PLDAModel(k=20)
        elif self.model_type == 'SLDA':
            mdl = tp.SLDAModel(k=num_topics, vars=['b' for _ in range(len(self.label_set))])

        mdl.add_corpus(self.corpus)


        for i in range(0, self.num_iters, 10):
            mdl.train(10)
            if verbose:
                print('Iteration: {}\tLog-likelihood: {}'.format(i, mdl.ll_per_word))

        if verbose:
            mdl.summary()

        self.lda_model = mdl
        model_path = os.path.join(model_dir_path, 'lda_model.bin')
        mdl.save(model_path)
        # self.save(model_dir_path)

        if self.model_type == 'LLDA':
            self.label_set = mdl.topic_label_dict

    def print_topics(self):
        mdl = self.lda_model
        # labels = mdl.topic_label_dict
        labels = self.label_set
    
        for k in range(mdl.k):
            print('Top 10 words of topic #{}'.format(k))
            if k<len(labels):
                print(labels[k])
            topic_words = [tup[0] for tup in mdl.get_topic_words(k, top_n=10)]
            print(topic_words)

    def predict_labels(self, document_data):

        ldamodel = self.lda_model
        corpus = self.corpus

        topic_df = pd.DataFrame()

        if self.model_type == 'SLDA':
            inferred, _ = self.lda_model.infer(self.corpus)
            # print(inferred)
            preds = self.lda_model.estimate(inferred)

            pred_labels = []
            pred_scores = []
            topic_words_per_doc = []

            for scores in preds:
                topic_num = np.argmax(scores)
                pred_scores.append(scores[topic_num])
                pred_labels.append(self.label_set[topic_num])
                # pred_labels.append(topic_num)
            #     print(pred, scores[i], label_set[i])
                topic_words = ', '.join([tup[0] for tup in ldamodel.get_topic_words(topic_num, top_n=10)])
                topic_words_per_doc.append(topic_words)


    #         topic_dist, ll = ldamodel.infer(corpus)
    #         for i, doc in enumerate(inferred):
    # #             row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
    #             topic_num, topic_pct = doc.get_topics(top_n=1)[0]

    #             pred_scores.append(topic_pct)
    #             pred_labels.append(self.label_set[topic_num])
        
    #             # get topic words
    #             topic_words = ', '.join([tup[0] for tup in ldamodel.get_topic_words(topic_num, top_n=10)])
    #             topic_words_per_doc.append(topic_words)
                # topic_df = topic_df.append(pd.Series([
                #     int(topic_num), 
                #     topic_label,
                #     round(topic_pct,4), 
                #     topic_words,
                #     ]), ignore_index=True)

            topic_df['topic_model_prediction'] = pred_labels
            topic_df['topic_model_prediction_score'] = pred_scores
            topic_df['topic_keywords'] = topic_words_per_doc

            # return document_data

        elif self.model_type == 'LLDA':

            # Init output
            # sent_topics_df = pd.DataFrame()

            labels = ldamodel.topic_label_dict
            print('topic labels:', labels)

            # Get main topic in each document
            topic_dist, ll = ldamodel.infer(corpus)
            for i, doc in enumerate(topic_dist):
    #             row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
                topic_num, topic_pct = doc.get_topics(top_n=1)[0]
        
                # get topic words
                topic_words = ', '.join([tup[0] for tup in ldamodel.get_topic_words(topic_num, top_n=10)])
                if topic_num < len(labels):
                    topic_label = labels[topic_num]
                else:
                    topic_label = 'other'
                
                topic_df = topic_df.append(pd.Series([
                    int(topic_num), 
                    topic_label,
                    round(topic_pct,4), 
                    topic_words,
                    ]), ignore_index=True)
            
            # sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
            topic_df.columns = [
                'dominant_topic_id', 
                'topic_model_prediction', 
                'topic_model_prediction_score', 
                'topic_keywords']
            # display(sent_topics_df)
            print(document_data.shape, topic_df.shape)

            
            # return(topic_df)

        # Add original text to the end of the output
        merged = pd.concat([document_data, topic_df], axis=1)
        # y_true = merged['tag_1']
        # y_pred = merged['lda_pred']
        # print(sklearn.metrics.accuracy_score(y_true, y_pred))

        return merged


    def get_dominant_topic(self, document_data):
        # get dominant topic for each document, along with topic keywords

        ldamodel = self.lda_model
        corpus = self.corpus

        # Init output
        sent_topics_df = pd.DataFrame()

        labels = self.label_set
        print('topic labels:', labels)

        # Get main topic in each document
        topic_dist, _ = ldamodel.infer(corpus)
        for i, doc in enumerate(topic_dist):
#             row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
            topic_num, topic_pct = doc.get_topics(top_n=1)[0]
    
            # get topic words
            topic_words = ', '.join([tup[0] for tup in ldamodel.get_topic_words(topic_num, top_n=10)])
            if topic_num < len(labels):
                topic_label = labels[topic_num]
            else:
                topic_label = 'other'
            
            sent_topics_df = sent_topics_df.append(pd.Series([
                int(topic_num), 
                topic_label,
                round(topic_pct,4), 
                topic_words,
                ]), ignore_index=True)
          

        # sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
        sent_topics_df.columns = ['dominant_topic_id', 'dominant_topic', 'dominant_topic_percent', 'topic_keywords']
        # display(sent_topics_df)

        # Add original text to the end of the output
        sent_topics_df = pd.concat([document_data, sent_topics_df], axis=1)
        return(sent_topics_df)

    def label_document(self, i, label):

        self.corpus[i].metadata = [label]
    
