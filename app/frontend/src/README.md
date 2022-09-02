# Source code implemeting frontend functionality

This is where the code that controls the user interface lives! 

1. The various css and javascript files call upon the code in `Components/` and `Views` to render the app. 

2. `Components/` contains the codes for various modular components used in the app frontend, like what is shown when the user clicks a document in the list shown to actually view the text of the passage, either choose from exisiting label set or submit a new label, etc. These are the components that are called upon and used for building the full pages a user sees as they use the app. 

3. `Views/` - a user has various options to view the documents - they can view them as a list, and monitor progress in various ways. There is also the home screen where they select or upload the data they are going to annotate and specify some settings for the app. The main view is the dashboard where they would usually perform the annotations. 

(Files in the above two folders are generally names according to what component or view they implement)

---

Two of the major files to consider (also distilled in the docs in main_functionality.ipynb): 

`src/Views/Dashboard.js`

This is the main file responsible for fetching what is needed to render the dashboard, most importantly, this fetches the document clusters (grouped by topic) and the next document to highlight from the backend (see refreshData()), and for then highlighting the chosen document and scrolling to it (in labelDocument()).

`src/Components/DocumentsByTopic.jsx`

The above finally passes the document clusters and highlighted next document element, etc. to this file in order to actually render what the user sees for documents grouped by topic on the dashboard. 