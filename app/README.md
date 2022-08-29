# Document Annotation App

An interface for annotating documents, with the help of topic models and active learning.

Frontend is written in React, backend is written in Python (FastAPI).

![user interface](interface.png)

## Features

- View documents clustered by topic
- Add, delete and rename labels
- Active learning: we recommend the next document to label
- Label predictions


## Run

`docker compose up`

Documentation: https://docs.google.com/document/d/17RkE-zmmuZYC3jWuBitwCuxxpng6DebMe17he2OZFds/edit#heading=h.8ioudbs2b60t

To use a shell, for example to run a jupyter notebook:
```
docker exec -it ras-guidance-docs-server-1 bash
jupyter notebook --ip 0.0.0.0 --allow-root
```

## Detailed Setup

Need: `poetry`, `yarn`

Install dependencies:  
`poetry install`  

You need two windows: one for the frontend, one for the backend+reverse proxy

Window 1:
Run backend:  
`poetry shell` 
`uvicorn backend.server:app --host 0.0.0.0 --port 81 --reload`

Window 2:
`cd frontend`
`yarn install`
`yarn start`
Access app at http://localhost:3000/  


## Architecture

### Frontend
- `frontend/src/`
  - `Views/Dashboard.jsx`: the UI container
  - `App.jsx`


### Backend

`backend/server.py` - main server

