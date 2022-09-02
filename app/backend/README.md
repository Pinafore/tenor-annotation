# Houses the backend implementation - all the code for the backend functionality

1. `data/` will contain the data files once downloaded using `get_dataset.sh` in the app directory -- these files should never be put on git!

2. `annotation_session.py`: This file in app/backend/ is the most critical backend file, containing the major functions called upon during a user session. While supported by other files in the backend/ folder, this file defines the user settings, creates document groupings by topics, identifies the document the classifier is most uncertain about (for active-learning based recommendations), trains and updates the classifiers, and maintain the updated data texts, scores, labels, etc. Essentially, this is the file that contains the functions called upon by the frontend in most cases.

3. `topic_model_new.py`: contains the topic modeling implementation being used in the app (see `docs/main_functionality.ipynb` for how to swap it out with a new topic model).

4. `server.py`: Coordinates calls from the frontend to the backend functions, and serves to tightly couple the frontend and backend. Frontend calls are routed through this, so substantial changes like addinng function in annotation_session.py might have to go through here. 