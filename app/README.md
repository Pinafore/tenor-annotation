# Document Annotation App

Original Author: Andrew Mao

Some updated by: Pranav Goel

An interface for annotating documents, with the help of topic models and active learning.

Frontend is written in React, backend is written in Python (FastAPI). **Please view the documentation in the sibling directory `docs/` in order to better understand the core functionality and state of the app (as of September 2nd, 2022).** 

![user interface](interface.png)

## Features

- View documents clustered by topic
- Add, delete and rename labels
- Active learning: we recommend the next document to label
- Label predictions


## Run locally

Documentation: https://docs.google.com/document/d/1C53gVS8RIN50hX9Ek3GWnntPv5AFV8Gx6RWMewCJ5fM/edit?usp=sharing

**FIRST, download data by running `bash get_dataset.sh`**

Need: `poetry`, `yarn`

You need two windows: one for the frontend, one for the backend+reverse proxy

Window 1:
Run backend:  
````bash
poetry install
poetry shell
python -m spacy download en_core_web_sm
uvicorn backend.server:app --host 0.0.0.0 --port 81 --reload
````

Window 2:
````bash
cd frontend
yarn install
yarn start
````

Access app at http://localhost:3000/  

## Deploy on server

To deploy on server use `docker-compose up`

## Directory structure - main files

1. `backend` contains all the backend code, implementing all the computational techniques used. The major file containing the core functions is annotation_session.py, supported by all the other files. 

2. `frontend` contains all the frontend code and instructions for getting familiar with React (the framework implemeting the frontend of the app).

3. `Dockerfile` can be modified for adding things for deploying the app through docker-compose

4. `get_dataset.sh` is the bash script that should get run first before the app is started in order to get the data used as the preloaded option for the app. 

## Core Architecture pieces:

### Frontend
- `frontend/src/`
  - `Views/Dashboard.jsx`: the UI container
  - `App.jsx`


### Backend

`backend/server.py` - main server

