# main class for document annotation

from datetime import date, datetime
import pickle
import os
import time
import shutil
import random
import threading

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from modAL.models import ActiveLearner
from modAL.uncertainty import classifier_uncertainty

# from backend.topic_model import TopicModel
from backend.topic_model_new import TopicModel



DEFAULT_SETTINGS = {
    "passage_chunk_max_words": 100,
    "use_topic_model":True,
    "topic_model_settings": {
        "model_type":"LLDA",
        "min_num_topics":10, # num topics will grow if there's more labels
        "num_iters": 100,
    },

    "tfidf_min":2,
    "tfidf_max":90,
    "use_active_learning":True,
    "classifier_model":"logistic regression", # naive bayes or logistic regression
    # "classifier_model":"naive bayes", # naive bayes or logistic regression
    "batch_update_freq": 3, # to update models
    "confidence_threshold": 0.8,
    "num_top_docs_shown": -1,
    "sort_docs_by": "uncertainty",
    "sample_size": 500
}
ACTIVE_LEARNER_MIN_DOCS = 3 # min # docs to start active learner


# DATA_PATH = '/fs/clip-quiz/amao/Github/document_annotation_app/backend/data'
DATA_PATH = 'backend/data'

class AnnotationSession():
    
    def __init__(self, username):

        self.username = username
        self.base_dir = f'backend/data/users/{self.username}'

        # make dir if it doesn't exist
        # if not os.path.exists(self.base_dir):
        #     os.makedirs(self.base_dir)

        self.dataset_name = 'none'
        self.preprocess_step = 'waiting for data'


        self.document_data = None
        self.topic_model = None
        self.labels = []
        
        # text features
        self.tfidf = None # vectorizer
        self.corpus_features = None

        # classifier and active learner
        self.learner = None
        self.active_learner_started = False

        # for the active learner
        self.minibatch_features = []
        self.minibatch_labels = []

        # user actions
        self.actions = []

        self.settings = DEFAULT_SETTINGS

        self.status = 'none' # status when updating data

    ## stats
    def get_num_labelled_docs(self):
        return self.document_data['manual_label'].notnull().sum()
#         print(f"number of labelled docs: {self.document_data['manual_labels'].notnull().sum()}")
    
    def get_statistics(self):

        if self.dataset_name == 'none': # no data
            return {}

        df = self.document_data
        num_docs = self.document_data.shape[0]
        num_labelled_docs = self.document_data['manual_label'].notnull().sum()
        num_unique_labels_used = self.document_data['manual_label'].notnull().unique().shape[0]
        num_labels = len(self.labels)
        num_confident_predictions = df[df['prediction_score']>=self.settings['confidence_threshold']].shape[0]
        num_docs_per_label = dict(self.document_data['manual_label'].value_counts())
        num_docs_per_label = {k:int(v) for k,v in num_docs_per_label.items()}
        for label in self.labels:
            if label not in num_docs_per_label:
                num_docs_per_label[label] = 0

        stats = {
            'dataset_name': self.dataset_name,
            'num_docs': num_docs,
            'num_labelled_docs': int(num_labelled_docs), 
            'num_unique_labels_used': num_unique_labels_used,
            'num_labels': num_labels,
            'num_confident_predictions': num_confident_predictions,
            'num_docs_per_label': num_docs_per_label,
            }
        print('stats', stats)
        return stats

    # def get_num_docs_per_label(self):
    #     return dict(self.document_data['manual_label'].value_counts())
    
    # def get_topics(self):
    #     # return self.topic_model.lda_model.print_topics()
    #     topic_words = []
    #     topic_weights = []
    #     for index, topic in self.topic_model.lda_model.show_topics(formatted=False, num_words=30):
    #         topic_words.append([w[0] for w in topic])
    #         topic_weights.append([w[1] for w in topic])

    #         # print('Topic: {} \nWords: {}'.format(idx, [w[0] for w in topic]))
    #     return topic_words, topic_weights

    def get_documents(self, limit):
        '''gets document metadata'''
        print('getting docs, limit = ' + str(limit))

        columns = [
            'doc_id',
            'text',
            'source',
            'manual_label',
            'predicted_label',
            'prediction_score',
            'uncertainty_score']

        if limit == -1:
            documents = self.document_data[columns].rename(columns={'doc_id':'id'}).fillna('')
        else:
            documents = self.document_data[columns].rename(columns={'doc_id':'id'}).head(limit).fillna('')

        res = documents.to_dict(orient='records')
        return res

    def get_document_clusters(self, group_size: int, sort_by: str):
        '''
        returns dict(document_clusters, doc_to_highlight)
        document_clusters: documents clustered by their dominant topic
        doc_to_highlight: id of the document to highlight, if any

        group_size: number of docs per cluster
        sort_by: uncertainty_score or prediction_score

        IMPORTANT FUNCTION (gets called on to get documents grouped by topics as well as the next document to highlight)
        '''
        
        columns = [
            'doc_id',
            'text',
            'source',
            'manual_label',
            'predicted_label',
            'prediction_score',
            'uncertainty_score',
            'previous_passage',
            'next_passage',
            'dominant_topic_percent',
            ]

        print('sort_by', sort_by)
        if sort_by == 'uncertainty':
            sort_by = 'uncertainty_score'
        elif sort_by == 'confidence':
            sort_by = 'prediction_score'

        document_clusters = []
        # print(self.document_data['dominant_topic'].head())
        # groupby = self.document_data.groupby('dominant_topic_id')
        # print(self.document_data)
        # print(self.document_data['manual_label'].isnull())
        '''
        if self.settings['use_active_learning'] and self.get_num_labelled_docs() >= ACTIVE_LEARNER_MIN_DOCS: # active learning is active
            groupby = self.document_data[self.document_data['manual_label'].isnull()].groupby('dominant_topic_id')
            print('\n --- ACTIVE LEARNING BLOCK OF GROUP BY --- \n')
        else:
            groupby = self.document_data[self.document_data['manual_label'].isnull()].groupby('dominant_topic_id')
            print('\n --- NOT ACTIVE LEARNING BLOCK OF GROUP BY --- \n')
            # groupby = self.document_data.groupby('topic_model_prediction')
        '''
        groupby = self.document_data[self.document_data['manual_label'].isnull()].groupby('dominant_topic_id')
        most_uncertain_docs = []

        
        # topic_labels = self.topic_model.lda_model.topic_label_dict
        topic_labels = self.topic_model.label_set
        
        print('topic labels:', topic_labels)
        #print('--- DOC DATA BEFORE GROUPING ---')
        #print(self.document_data.info())
        for topic_id, group in groupby:

            # get the most uncertain documents
            sorted_group = group.sort_values(sort_by, ascending=False)
            if group_size == -1:
                doc_ids = sorted_group['doc_id']
            else:
                doc_ids = sorted_group['doc_id'].head(group_size)

            # print(self.document_data.iloc[doc_ids].shape)
            documents = self.document_data.iloc[doc_ids][columns].fillna('')
            most_uncertain_docs.append(documents.head(1))

            documents = documents.to_dict(orient='records')

            # get the topic label, if any
            topic_id = int(topic_id)
            if topic_id < len(topic_labels):
                topic_label = topic_labels[topic_id]
            else:
                topic_label = "None"

            # print(group['manual_label'])
            # num_labeled_docs = int((group['manual_label'] == topic_label).sum())
            num_labelled_docs = int((group['manual_label'].notnull()).sum())
            num_docs = int(group.shape[0])
            # print('topic_label', topic_label)
            # print(num_labeled_docs)

            document_clusters.append({
                # 'topic_id': topic_id, 
                'topic_words': group.iloc[0]['topic_keywords'], 
                'documents': documents,
                'topic_label': topic_label,
                'num_labelled_docs': num_labelled_docs, 
                'num_docs': num_docs,
                # how many docs in the group have actually been labelled? 
                })

        #if self.settings['use_active_learning'] and self.get_num_labelled_docs() >= ACTIVE_LEARNER_MIN_DOCS:
            #doc_to_highlight = int(random.choice(most_uncertain_docs)['doc_id'])
        #else:
            #doc_to_highlight = int(random.choice(most_uncertain_docs)['doc_id'])#random.choice(list(self.document_data[self.document_data['manual_label'].isnull()].index))
        #print('Most uncertain docs ----')
        #print(pd.concat(most_uncertain_docs, 0).info())
        doc_to_highlight = int(pd.concat(most_uncertain_docs, 0).sort_values(sort_by, ascending=False).head(1)['doc_id']) #int(random.choice(most_uncertain_docs)['doc_id'])
        print(type(self.document_data['uncertainty_score']))
        print(self.document_data['uncertainty_score'].describe())
        print('Doc to highlight = ' + str(doc_to_highlight))
        return {'document_clusters': document_clusters, 'doc_to_highlight': doc_to_highlight}


    ### labels

    def add_label(self, label):
        if label not in self.labels:
            self.labels.append(label)

        self.record_action('add_label', label)

    def delete_label(self, label):
        if label in self.labels:
            self.labels.remove(label)

            # mask = self.document_data['manual_label'].isin(self.labels)
            mask = self.document_data['manual_label'] == label

            # clear annotations: label, scores
            # self.document_data.loc[mask, 'manual_label'] = None
            # self.document_data.loc[mask, ['manual_label', 'predicted_label', 'prediction_score', 'uncertainty_score']] = None
            self.document_data.loc[:, ['predicted_label', 'prediction_score', 'uncertainty_score']] = None
            self.document_data.loc[mask, ['manual_label']] = None

            # update topic model
            self.train_topic_model()

            # if there's <=1 labels, delete classifier 
            if len(self.labels) <= 1:
                self.learner = None
                self.active_learner_started = False
                return

            # else, refit classifier, get new scores
            if self.learner:
                df = self.document_data
                df = df[df['manual_label'].notnull()]
                # df = df[mask]
                X = self.tfidf.transform(df['text'])
                y = df['manual_label']
                # print(X)
                self.learner.fit(X,y)

                self.update_document_metadata()

        self.record_action('delete_label', label)


    def rename_label(self, old_label, new_label):
        if old_label in self.labels:
            self.labels.remove(old_label)
            self.labels.append(new_label)

            mask = self.document_data['manual_label'] == old_label
            self.document_data.loc[mask, 'manual_label'] = new_label

            # topic model
            self.train_topic_model()

            # refit classifier
            if self.learner:
                df = self.document_data
                df = df[df['manual_label'].isin(self.labels)]
                X = self.tfidf.transform(df['text'])
                y = df['manual_label']
                self.learner.fit(X,y)

        self.record_action('rename_label', {'old_label': old_label, 'new_label': new_label})

    ### topic model

    def train_topic_model(self):

        topic_model_path = self.base_dir
        self.status = 'training topic model...'

        # topic_model = TopicModel(model_type=DEFAULT_SETTINGS['topic_model'])
        topic_model = TopicModel(**self.settings['topic_model_settings'])

        
        document_data = self.document_data
        documents = document_data.text.values.tolist()
        labels = list(document_data.manual_label)
        topic_model.preprocess(documents, topic_model_path, labels=labels, fast=True)

        # self.preprocess_step = '(3/4) training topic model...'
        topic_model.train(topic_model_path)

        # add topic model metadata
        # self.preprocess_step = '(4/4) setting up database...'
        columns = ['dominant_topic_id', 'topic_model_prediction', 'topic_model_prediction_score', 'topic_keywords', 'dominant_topic_percent', 'dominant_topic']
        to_drop = []
        for col in columns:
            if col in document_data:
                to_drop.append(col)

        # document_data = document_data.drop(['dominant_topic_id','dominant_topic','dominant_topic_percent','topic_keywords'], axis=1)
        document_data = document_data.drop(to_drop, axis=1)

        merged = topic_model.get_dominant_topic(document_data)
        # merged = topic_model.predict_labels(document_data)

        self.document_data = merged

        self.topic_model = topic_model


    ### document actions

    # view document: text, context, predicted label and confidence, top topic(s)
    def view_document(self, doc_id):
        doc = self.document_data.loc[doc_id]
        print(doc)

    def initialize_classifier(self, text, labels):
        print('initializing active learner')
        self.status = 'starting active learning...'

        # setup vectorizer
        # text = self.document_data['text'].apply(str).tolist()
        self.tfidf = TfidfVectorizer(stop_words='english', lowercase=True, ngram_range=(1,2))
        self.corpus_features = self.tfidf.fit_transform(text)

        # df = self.document_data
        train_text = text[labels.notnull()]
        train_data = self.tfidf.transform(train_text)
        

        # self.minibatch_features = self.tfidf.transform(self.minibatch_features)
        # print(self.minibatch_features.shape, len(self.minibatch_labels))

        # self.minibatch_features = self.document_data['manual_label']
        # print(len(self.minibatch_features))
        # df = self.document_data
        # df = df[df['manual_label'].notnull()]
        # train_data = self.tfidf.transform(text)

        # train_labels = labels
        train_labels = list(labels.dropna())

        model = self.settings['classifier_model'].lower()
        print('model', model)
        if model == 'logistic regression':
            estimator=SGDClassifier('log')
        elif model == 'naive bayes':
            estimator=MultinomialNB()
        else:
            raise Exception("Invalid classifier model")

        self.learner = ActiveLearner(
            estimator=estimator,
            X_training=train_data, 
            y_training=train_labels
        )
        self.active_learner_started = True

    # label document: update models, update data with scores from the classifier. if batch finished: update classifier, topic model
    def label_document(self, doc_id, label, update_topic_model=True):
        #another IMPORTANT FUNCTION

        self.status = 'processing...'
        self.record_action('label_document', {'doc_id': doc_id, 'label': label})
        
        if label not in self.labels:
            self.labels.append(label)
        self.document_data.loc[doc_id,'manual_label'] = label

        # row = self.document_data.loc[doc_id]

        # # topic model update logic
        # # print('TOPIC MODEL', self.document_data.shape[0] % TOPIC_MODEL_UPDATE_FREQ)
        if update_topic_model and self.document_data['manual_label'].notnull().sum() % self.settings['batch_update_freq'] == 0: # retrain model
            print('retraining topic model...')
            # t = threading.Thread(target=train_topic_model, args=(self))
            t = threading.Thread(target=self.train_topic_model)
            t.start()
            # self.train_topic_model()

        # active learning logic
        print('active learning logic...')
        if self.settings['use_active_learning'] and self.get_num_labelled_docs() >= ACTIVE_LEARNER_MIN_DOCS: # active learning
            #print('\n --- ACTIVE LEARNING BEING USED BLOCK  --- \n')
            if not self.active_learner_started: # start active learning
                print('\n --- STARTING ACTIVE LEARNING  --- \n')
                df = self.document_data
                # df = df[df['manual_label'].notnull()]
                # train_data = self.tfidf.transform(list(df['text']))
                # train_labels = list(df['manual_label'])

                self.initialize_classifier(df['text'], df['manual_label'])

                # print('initializing active learner')
                # self.status = 'starting active learning...'

                # # setup vectorizer
                # text = self.document_data['text'].apply(str).tolist()
                # self.tfidf = TfidfVectorizer(stop_words='english', lowercase=True, ngram_range=(1,2))
                # self.corpus_features = self.tfidf.fit_transform(text)

                # # self.minibatch_features = self.tfidf.transform(self.minibatch_features)
                # # print(self.minibatch_features.shape, len(self.minibatch_labels))

                # # self.minibatch_features = self.document_data['manual_label']
                # # print(len(self.minibatch_features))
                # df = self.document_data
                # df = df[df['manual_label'].notnull()]
                # train_data = self.tfidf.transform(list(df['text']))
                # train_labels = list(df['manual_label'])

                # self.learner = ActiveLearner(
                #     estimator=SGDClassifier('log'),
                #     # estimator=MultinomialNB(),
                #     X_training=train_data, y_training=train_labels
                # )
                # self.active_learner_started = True
            else:
                #print('\n --- ACTIVE LEARNING: UPDATING CLASSIFIER  --- \n')
                print('updating classifier...')
                self.status = 'updating classifier...'

                query_idx = doc_id # doc id must equal query idx
                # print(type(self.corpus_features), type(self.corpus_features[query_idx]), type(label))
                # print(self.corpus_features.shape, self.corpus_features[query_idx].shape, [label])

                self.learner.teach(self.corpus_features[query_idx], [label]) 

                # multithreading bugs?
                # t = threading.Thread(target=self.update_classifier, args=(self.corpus_features[query_idx], label))
                # t.start()
            
            # update model predictions
            self.update_document_metadata()

            return "active_learning_update"
        else:
            return 'no_active_learning'

    def update_classifier(self, features, label):
        self.learner.teach(features, [label]) 


    def update_document_metadata(self):
        '''
        metadata from the classifier
        '''

        # print('self.corpus_features', self.corpus_features)
        uncertainty_score = classifier_uncertainty(self.learner, self.corpus_features)
        uncertainty_score = np.round(uncertainty_score, 3)
        self.document_data['uncertainty_score'] = uncertainty_score
        self.document_data['predicted_label'] = self.learner.predict(self.corpus_features)
        prediction_scores = self.learner.estimator.predict_proba(self.corpus_features)
        prediction_scores = prediction_scores.max(axis=1).round(3)
        self.document_data['prediction_score'] = prediction_scores

        # manually set uncertainty score for labelled docs
        self.document_data.loc[self.document_data['manual_label'].notnull(), 'uncertainty_score'] = 0

    '''
    THE BELOW FUNCTION IS NOT ACTUALLY GETTING USED RIGHT NOW (but keeping it in case the logical flow of the app code needs to be changed in order to call on a separate function to get the next document to label (right now, the app uses the second item returned by get_document_clusters() above in order to get the next document to highlight.
    # choose the next doc to annotate
    def get_next_document_to_label(self):
        #return int(self.document_data['uncertainty_score'].argmax())
    '''

    '''
    def get_next_document_to_label_random(self):
        print('\n --- NEXT DOC TO LABEL - RANDOM --- \n')
        z = random.choice(list(self.document_data[self.document_data['manual_label'].isnull()].index))
        print(type(z))
        print(z)
        return z#random.choice(list(self.document_data[self.document_data['manual_label'].isnull()].index))
    '''

    ### saving and loading

    def save_data(self):

        # save the annotations, the aux data (labels, actions), classifier, and session
        print('saving data...')
        username = self.username
        root_dir = self.base_dir
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        data_path = os.path.join(root_dir,'saved_annotations.csv')
        self.document_data.to_csv(data_path, mode='w+') 

        save_path = os.path.join(root_dir,'saved_data.pkl')
        data = {
            'labels': self.labels,
            'actions': self.actions,
            'dataset': self.dataset_name
        }
        with open(save_path, 'wb+') as f:
            pickle.dump(data, f)

        save_path = os.path.join(root_dir,'classifier.pkl')
        with open(save_path, 'wb+') as f:
            pickle.dump(self.learner, f)

        # session_path = os.path.join(root_dir,'saved_session.pkl')
        # with open(session_path, 'wb+') as f:
        #     pickle.dump(self, f)

    def load_data(self, load_preset=False):
        
        root_dir = self.base_dir
        # root_dir = f'/fs/clip-quiz/amao/Github/document_annotation_app/backend/data/{self.username}'
        # self.base_dir = root_dir

        if not os.path.exists(os.path.join(root_dir,'saved_annotations.csv')):
            print('no saved data. exiting...')
            return

        # found a directory, assumes it has saved data
        print('found saved data. loading...')
        data_path = os.path.join(root_dir,'saved_annotations.csv')
        self.document_data = pd.read_csv(data_path)
        # self.document_data['source'] = None

        # topic model
        # self.topic_model = TopicModel(
        #     model_type=DEFAULT_SETTINGS['topic_model'], 
        #     num_topics=self.settings['min_num_topics'],
        #     num_iters=self.settings['topic_model_num_iters'],
        #     )
        self.topic_model = TopicModel(**self.settings['topic_model_settings'])

        self.topic_model.load_model(self.base_dir)

        # labels, actions, classifier
        save_path = os.path.join(root_dir,'saved_data.pkl')
        with open(save_path, 'rb') as f:
            data = pickle.load(f)
            self.labels = data['labels']
            self.actions = data['actions']
            self.dataset_name = data['dataset']

        save_path = os.path.join(root_dir,'classifier.pkl')
        with open(save_path, 'rb') as f:
            self.learner = pickle.load(f)

        # vectorizer
        text = self.document_data['text'].apply(str).tolist()
        self.tfidf = TfidfVectorizer(stop_words='english', lowercase=True, ngram_range=(1,2))
        self.corpus_features = self.tfidf.fit_transform(text)

        # session_path = os.path.join(root_dir,'saved_session.pkl')
        # with open(save_path, 'rb') as f:
        #     data = pickle.load(f)

    def record_action(self, action_name, data):
        record = {'timestamp': datetime.now(), 'name': action_name, 'data': data}
        self.actions.append(record)

    def reset(self):

        # just clear the annotations, keep the loaded data, topic model
        
        self.document_data[['manual_label',
            'predicted_label',
            'prediction_score',
            'uncertainty_score']] = None

        self.labels = []

        self.learner = None
        self.active_learner_started = False

        self.minibatch_features = []
        self.minibatch_labels = []

        # user actions
        self.actions = []
        self.settings = DEFAULT_SETTINGS

        shutil.rmtree(self.base_dir)


    def hard_reset(self):

        self.reset()
        # delete files
        shutil.rmtree(self.base_dir)
        


    def set_settings(self, settings):
        for k,v in settings.items():
            self.settings[k] = v

        print('settings', self.settings)

    def get_docs_grouped_by_label(self):
        groups = {}
        for label, group in self.document_data.groupby('manual_label'):
            groups[label] = group.fillna('').to_dict(orient='records')
        for label in self.labels:
            if label not in groups.keys():
                groups[label] = []
        print('groups', groups.keys())
        return groups

    def setup_data(
        self, 
        document_data, 
        dataset_name, 
        import_annotations=False, 
        use_preset=False,
        train_topic_model=True,
        preset_topic_model_path=DATA_PATH,
        ):
        '''
        Setup annotation session: train topic model and setup database.
        '''
        
        self.dataset_name = dataset_name
        self.preprocess_step = '(2/4) preprocessing data...'

        # train topic model
        # root = f'/fs/clip-quiz/amao/Github/document_annotation_app/backend/data/{self.username}'
        root = self.base_dir
        # make dir if it doesn't exist
        if not os.path.exists(root):
            os.makedirs(root)

        # topic_model_path = os.path.join(root, f"data/{self.username}")
        topic_model_path = root
        
        merged = document_data
        # .topic_model = TopicModel(
        #     model_type=self.settings['topic_model'], 
        #     num_topics=self.settings['min_num_topics'],
        #     num_iters=self.settings['topic_model_num_iters'],
        #     )
        topic_model = TopicModel(**self.settings['topic_model_settings'])


        if use_preset:
            print('using preset topic model...')
            topic_model_path = preset_topic_model_path
            topic_model.load_model(topic_model_path) 
        elif train_topic_model:
            # topic_model = TopicModel()
            documents = document_data.text.values.tolist()

            print('topic_model_path', topic_model_path)
            if import_annotations:
                print('importing annotations...')
                labels = list(document_data.label)
                topic_model.preprocess(documents,topic_model_path,labels=labels)
            else:
                topic_model.preprocess(documents,topic_model_path)

            self.preprocess_step = '(3/4) training topic model...'
            topic_model.train(topic_model_path)

            self.preprocess_step = '(4/4) setting up database...'
            merged = topic_model.get_dominant_topic(document_data)
            # merged = topic_model.predict_labels(document_data)

            topic_model.save(root)

                
        self.topic_model = topic_model

        # add columns
        merged['doc_id'] = range(merged.shape[0])
        if 'source' not in merged:
            merged['source'] = None
        
        merged['manual_label'] = None
        merged['predicted_label'] = None
        merged['prediction_score'] = None
        merged['uncertainty_score'] = None

        merged['previous_passage'] = merged['text'].shift(-1)
        merged['next_passage'] = merged['text'].shift(1)

        self.document_data = merged

        if import_annotations:
            self.document_data['manual_label'] = labels
            # self.initialize_classifier(merged['text'], labels)
            self.initialize_classifier(merged['text'], self.document_data['manual_label'])
            self.update_document_metadata()

            # set labels
            self.labels = list(set(labels))
            if np.nan in self.labels:
                self.labels.remove(np.nan)

        self.save_data() # does this save the topic model?

        self.preprocess_step = 'finished'

        return True

    def get_preprocessing_status(self):
        return self.preprocess_step

    def get_status(self):
        return self.status
