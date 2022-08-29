from fastapi import FastAPI, Depends, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import FileResponse

import os
import traceback
import pickle

import pandas as pd

# import string
# import re
# import nltk

# from backend.topic_model import TopicModel
from backend.annotation_session import AnnotationSession
from backend import security
from backend.security import get_current_user

from datetime import date
import shutil
import time

# from multiprocessing import Process, Queue
import threading

# default data
# root_path = "/fs/clip-quiz/amao/Github/alto-boot"
# TLP_DOCUMENTS_PATH = os.path.join("data", "nist_ras_documents_cleaned.csv")
TLP_DOCUMENTS_PATH = os.path.join("backend", "data/nist_ras_documents_cleaned.csv")

# data_path = "data/nist_ras_cleaned.csv"
# document_df = pd.read_csv(document_path)


# cache for storing sessions.
sessions = dict()

def get_session(current_user: str):
    print('current user: ', current_user)

    if current_user not in sessions:
        print('session not in cache')
        session = AnnotationSession(current_user)
        sessions[current_user] = session

        # # saved state from disk
        session.load_data()
        # state = db.get_user_state(current_user)
        # if state:
        #     print('loading state from disk...')
        #     game_object.load(state)
        #     # print('question number', game_object.state['question_number'])
    # else:
    #     print('loading cached session...')
        
    return sessions[current_user]

def delete_session(current_user: str):
    del sessions[current_user]



# app setup

app = FastAPI()
# db = Database()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex="https://.*\.ngrok\.io",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(security.router, prefix="/token")
# app.include_router(data_server.router)


# endpoints

@app.get("/")
def read_root():
    return {"Hello": "World"}

# get player info from database
@app.get("/get_player_info")
async def get_player_info(current_user: str = Depends(get_current_user)):
    print(f'username: {current_user}')
    return current_user

# annotation stats
@app.get("/get_statistics")
async def get_statistics(current_user: str = Depends(get_current_user)):
    session = get_session(current_user)
    return session.get_statistics()

@app.get("/get_documents")
async def get_documents(limit: int = -1, current_user: str = Depends(get_current_user)):
    session = get_session(current_user)
    return session.get_documents(limit)

@app.get("/get_document_clusters")
async def get_document_clusters(sort_by: str = 'uncertainty', group_size: int = 5, current_user: str = Depends(get_current_user)):
    session = get_session(current_user)
    clusters = session.get_document_clusters(group_size, sort_by)
    return clusters

# get dictionary of docs for each label
@app.get("/get_documents_grouped_by_label")
async def get_documents_grouped_by_label(current_user: str = Depends(get_current_user)):
    session = get_session(current_user)
    return session.get_docs_grouped_by_label()



### labels

@app.get("/get_labels")
async def get_labels(current_user: str = Depends(get_current_user)):
    session = get_session(current_user)
    print('labels', session.labels)
    return session.labels

@app.post("/add_label")
async def add_label(label: str, current_user: str = Depends(get_current_user)):
    session = get_session(current_user)
    session.add_label(label)
    session.save_data()

@app.post("/delete_label")
async def delete_label(label: str, current_user: str = Depends(get_current_user)):
    session = get_session(current_user)
    session.delete_label(label)
    session.save_data()

@app.post("/rename_label")
async def rename_label(old_label: str, new_label: str, current_user: str = Depends(get_current_user)):
    session = get_session(current_user)
    session.rename_label(old_label, new_label)
    session.save_data()

# @app.get("/get_topics")
# async def get_topics():
#     topic_words, topic_weights = session.get_topics()
#     return {'topic_words': topic_words, 'topic_weights': topic_weights}


@app.post("/label_document")
async def label_document(doc_id: int, label: str, current_user: str = Depends(get_current_user)):
    session = get_session(current_user)
    label_resp = session.label_document(doc_id, label)
    session.save_data()
    resp = {'status': label_resp}
    # if label_resp == "no_active_learning":
    #     return resp
    # else:
    #     resp['next_doc_id_to_label'] = session.get_next_document_to_label()
    #     print('resp', resp)
    #     return resp

    if label_resp == "active_learning":
        resp['next_doc_id_to_label'] = session.get_next_document_to_label()
    # # update topic model
    # print('retraining topic model...')
    # # t = threading.Thread(target=train_topic_model, args=(session))
    # t = threading.Thread(target=session.train_topic_model)
    # t.start()

    return resp



@app.post("/reset")
async def reset(current_user: str = Depends(get_current_user)):
    session = get_session(current_user)
    session.reset()

# settings
@app.get("/get_settings")
async def get_settings(current_user: str = Depends(get_current_user)):
    session = get_session(current_user)
    return session.settings

@app.post("/set_settings")
async def set_settings(settings: dict, current_user: str = Depends(get_current_user)):
    session = get_session(current_user)
    session.set_settings(settings)

# upload csv file
# async def upload_data(file: bytes = File(...)):
@app.post("/upload_csv")
async def upload_data(file: UploadFile = File(...), current_user: str = Depends(get_current_user)):
    session = get_session(current_user)
    print("filename",file.filename)

    try:
        df = pd.read_csv(file.file)
        print(df)
        res = session.setup_data(df, file.filename)
        return res
    except pd.errors.ParserError:
        print('file parse error')
        return {'error': 'parse_error'}
    except Exception as e:
        print('processing error')
        return {'error': e}


# process uploaded data
def process_data(file, session, file_type, import_annotations):

    try:
        session.preprocess_step = '(1/4) extracting files...'

        if file_type == "csv":
            df = pd.read_csv(file.file)
        elif file_type == "zip":

            # unzip
            extract_dir = 'tmp'
            if not os.path.exists(extract_dir):
                os.makedirs(extract_dir)
            
            with open('tmpfile.zip','wb+') as f:
                f.write(file.file.read())
            print('unzipping...')
            shutil.unpack_archive('tmpfile.zip', extract_dir)

            # read files
            print('reading...')
            documents = []
            document_titles = []
            doc_ids = []
            doc_id = 0
            # PASSAGE_MAX_WORDS = 200
            for docname in os.listdir(extract_dir):
                # words = f.split('_')
                # doc_id = int(words[-1])
                # title = '_'.join(words[:-1])

                # doc_ids.append(doc_id)
                # doc_id += 1
                # document_titles.append(docname)

                filepath = os.path.join(extract_dir, docname)
                
                # with open(filepath, 'r') as f:
                #     documents.append(f.read())
                
                with open(filepath, 'r') as f:
                    # documents.append(f.read())
                    passage = []
                    num_words = 0
                    for line in f:
                        words = line.split()
                        passage += words
                        if len(passage) >= session.settings['passage_chunk_max_words']:
                            documents.append(' '.join(passage))
                            doc_ids.append(doc_id)
                            doc_id += 1
                            document_titles.append(docname)

                            passage = []
                    if passage: 
                        documents.append(' '.join(passage))
                        doc_ids.append(doc_id)
                        doc_id += 1
                        document_titles.append(docname)

            # print(len(documents), len(document_titles)
            print('number of passages: ', doc_id)


            

            df = pd.DataFrame({
                'id': doc_ids, 
                'source': document_titles, 
                'text': documents}).sort_values('id')
                # .set_index('id')
            # df = pd.read_csv(file.file)
            # print(df)
        
            # session.setup_data(df, file.filename)

        else:
            raise Exception("unsupported file format")
        


        session.setup_data(df, file.filename, import_annotations=import_annotations)
        print('finished processing new data!')
        # return res

    except Exception as e:
        print('error processing new data!', e)
        verbose_error = traceback.format_exc()
        print(verbose_error)
        session.preprocess_step = 'error! message: ' + str(verbose_error)

    finally:
        # cleanup
        print('cleaning up...')
        shutil.rmtree(extract_dir)
        os.remove('tmpfile.zip')

# process uploaded data in a new thread
@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...), file_type = None, import_annotations: bool = False, current_user: str = Depends(get_current_user)):
    session = get_session(current_user)
    print("filename",file.filename)

    t = threading.Thread(target=process_data, args=(file,session,file_type,import_annotations))
    t.start()
    # p.join()

    print('done')
    return

    # except pd.errors.ParserError:
    #     print('file parse error')
    #     return {'error': 'parse_error'}
    # except Exception as e:
    #     print('processing error', e)
    #     return {'error': e}




@app.post("/use_preloaded_data")
async def use_preloaded_data(current_user: str = Depends(get_current_user)):

    def process_csv(filepath, session):
        df = pd.read_csv(filepath)
        res = session.setup_data(df, filepath, use_preset=True)
        return res

    session = get_session(current_user)
    
    # print('starting thread...')
    # t = threading.Thread(target=process_csv, args=(TLP_DOCUMENTS_PATH,session))
    # t.start()

    print(os.listdir('backend/data'))

    process_csv(TLP_DOCUMENTS_PATH,session)
    print('document_data', session.document_data)

    return


def f(q):
    for i in range(5):
        print('processing', i)
        q.put(i)
        time.sleep(1)

# q = Queue()

@app.post("/upload_data_dummy")
async def upload_data_dummy(current_user: str = Depends(get_current_user)):
    print('processing...')
    
    p = threading.Thread(target=f)
    p.start()
    # print(q.get())    # prints "[42, None, 'hello']"
    # p.join()

    print('done')
    return

# @app.post("/ingest_data")
# async def ingest_data(current_user: str = Depends(get_current_user)):
#     session = get_session(current_user)
#     document_df = pd.read_csv(document_path)
#     res = session.setup_data(document_df)
#     return res

@app.get("/get_preprocessing_status")
async def get_preprocessing_status(current_user: str = Depends(get_current_user)):
    session = get_session(current_user)
    res = session.get_preprocessing_status()
    return res

@app.get("/get_status")
async def get_status(current_user: str = Depends(get_current_user)):
    session = get_session(current_user)
    res = session.get_status()
    return res

@app.get("/export_data")
async def export_data(current_user: str = Depends(get_current_user)):
    session = get_session(current_user)
    filepath = f'backend/data/{current_user}/exported_annotations.csv'
    # session.document_data.to_json(filepath, orient='records', lines=True)
    session.document_data.to_csv(filepath)

    return FileResponse(filepath, media_type='application/octet-stream', filename='exported_annotations.csv')

