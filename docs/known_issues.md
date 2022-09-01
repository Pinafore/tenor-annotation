### All issues are currently noted under the current default settings (unless noted otherwise) and on the Dashboard one gets to after choosing the dataset and selecting "Start Annotation Session" (and sees passages grouped by topics, the main place where users read passages and create labels and label them). To get a general flow and tutorial for using the app, see the google doc linked in the README in app/. 

# 1. The next document to label does not always get highlighted.

*Intended behavior:* Ideally, from the beginning and after every instance of the user labeling a passage/document, the next document that a user should move on to labeling should get highlighted by a red box and should get scrolled to. 

*Actual behavior:* But this sometimes happen and sometimes just does not! And sometimes the app scrolls below the highlighted doc (i.e. the next document is highlighted by a red box but the app scrolls way past it). Sometimes no doc is highlighted at all, even though the app is able to fetch the next document to highlight and it is definitely present on the dashboard. 

*Probable relevant code pieces*: 

- `frontend/src/Views/Dashboard.js`: This is the main file responsible for fetching what is needed to render the dashboard, so this fetches the document clusters (grouped by topic) and the next document to highlight from the backend (see refreshData()), and for then highlighting the chosen document and scrolling to it (in labelDocument()). 

- `frontend/src/Components/DocumentsByTopic.jsx`: The above finally passes the document clusters and highlighted next document element, etc. to this file in order to actually render what the user sees for documents grouped by topic on the dashboard. While the current code passes the document clusters and other info from Dashboard.js to this file which should sync up the document groupings and document data being used in the frontend, there does seem to be some issue where there is a disconnect between the document data or clusters created at backend and what is being used on the frontend (see Issue 2 below). 

*Other possibly relevant considerations:

User settings **batch_update_freq** and **num_top_docs_shown** values might interact with this issue and be playing a role. Descriptions of settings can be found in app/backend/annotation_session.py; and customized_settings.md documents where modifications need to happen in order to modify the user setting options and default values. 

# 2. When group_size or num_top_docs_shown (setting) is set to a number other than -1 (i.e. show some top documents for every topic, not all), app frequently fails to find the next document to label (so there is nothing to highlight because when fetching the next document by ID, it gets null)

Note: This is a different, but perhaps connected, problem from issue 1. By default, num_top_docs_shown is set to -1, and in that case, there is no failure in locating and fetching the next doc to label, but only glitches as noted in issue 1. But num_top_docs_shown = -1 is not the best choice from a UX point of view - at any given instance, having all documents for a particular topic being shown is unnecessary and a lot. Only certain top documents like 5/10/20 per topic group should be shown. But when that is enabled by setting this kind of value for num_top_docs_shown, we do get the desired number of docs shown per topic, but the app then fails to find the next document to label it got from the backend (document.getElementByID(next_doc) where next_doc is the doc ID to highlight next returns null).

*Probable relevant code pieces*: 

- `frontend/src/Views/Dashboard.js`
- `frontend/src/Components/DocumentsByTopic.jsx`
- `app/backend/annotation_session.py`

*Speculation*: It seems like there is a disconnect between the document clusterings at backend and frontend somehow. Note that `frontend/src/Components/DocumentsByTopic.jsx` receives the document clusters and next doc info from `frontend/src/Views/Dashboard.js` to render it. `frontend/src/Views/Dashboard.js` is the critical frontend piece. That code receives the document clusters and net doc to label in its **refreshData** function by calling **get_document_clusters()** function located in `app/backend/annotation_session.py`. The other critical function in `frontend/src/Views/Dashboard.js` is **labelDocument()**. 

*Intended Behavior*: It should never be the case that the app is unable to fetch the next document to label because everything should be working with the same collection and groupings of documents. 

### NOTE: Other potentially relevant functions for these two issues are `update_document_metadata()` and `label_documents()` in `app/backend/annotation_session.py`.