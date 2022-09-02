# pytest tests

# test the annotation API

import pandas as pd
import numpy as np
import string
import re
# import nltk

import sys
sys.path.append("backend")
# sys.path.append("/fs/clip-quiz/amao/Github/document_annotation_app/backend")

from topic_model import TopicModel
from annotation_session import AnnotationSession


import os
import pandas as pd

# read data
# root_path = "/fs/clip-quiz/amao/Github/alto-boot"
TLP_DOCUMENTS_PATH = 'backend/data/nist_ras_documents_cleaned.csv'
document_data = pd.read_csv(TLP_DOCUMENTS_PATH)



# test creating session, importing annotations, labelling data
def test_session_fully_labelled(): 

    # create session
    session = AnnotationSession('test_user')
    session.setup_data(document_data, 'TLP', import_annotations=True)
    # session.setup_data(partially_labelled_data, 'TLP')

    labels = session.document_data.label

    for i in range(5):
        print(i, labels[i])
        session.label_document(i, labels[i], update_topic_model=False)
    
    # update the topic model
    session.label_document(i, labels[i], update_topic_model=True)

    session.delete_label(labels[0])

# test with unlabelled data
def test_session_unlabelled(): 

    # create session
    session = AnnotationSession('test_user')
    session.setup_data(document_data, 'TLP')
    # session.setup_data(partially_labelled_data, 'TLP')

    labels = session.document_data.label

    for i in range(5):
        print(i, labels[i])
        session.label_document(i, labels[i], update_topic_model=False)
    
    # update the topic model
    session.label_document(i, labels[i], update_topic_model=True)

    session.delete_label(labels[0])


# test with partial labels
def test_session_partial_labels(): 

    # create session
    session = AnnotationSession('test_user')
    session.setup_data(document_data, 'TLP')
    # session.setup_data(partially_labelled_data, 'TLP')

    labels = session.document_data.label

    for i in range(5):
        print(i, labels[i])
        session.label_document(i, labels[i], update_topic_model=False)
    
    # update the topic model
    session.label_document(i, labels[i], update_topic_model=True)

    session.delete_label(labels[0])

# test importing unlabelled data
# def test_session_2():

#     # create session
#     session = AnnotationSession('test_user')
#     # def setup_data(self, document_data, dataset_name, use_preset=False):
#     # session.setup_data(document_data, 'TLP', use_preset=True)
#     session.setup_data(document_data, 'TLP', import_annotations=True)
#     # session.setup_data(partially_labelled_data, 'TLP')

#     # session.load_data()
#     # session.reset()

#     # simulate adding some data
#     # labels = ['one','two','three','one','two']
#     # labels = session.document_data.tag_1
#     labels = session.document_data.label

#     # for label in labels:
#     #     session.add_label(label)
#     # for i in range(len(labels)):
#     #     print(i, labels[i])
#     #     session.label_document(i, labels[i], update_topic_model=False)

#     # topic_label_dict = session.topic_model.lda_model.topic_label_dict

#     for i in range(5):
#         print(i, labels[i])
#         session.label_document(i, labels[i], update_topic_model=False)
        
#     session.label_document(i, labels[i], update_topic_model=True)
#     # session.delete_label('one')

