### How to modify what settings are shown in the interface or used in the code - where to add, delete, or otherwise modify settings including their default values: 

There are two places to make such changes to control the settings for the app: 

1. app/backend/annotation_session.py:  Line 26 after all modules are imported, also explains the settings. 

2. app/frontend/src/Components/Settings.jsx: Line 22 - this controls what is shown to the user (and interacts with settings defined at the backend) after the login screen before annotation begins (the Home screen). 