### Assuming known issues are resolved, these are the planned various next steps (not comprehensive): 

1. (Tech, good dev practices) **Incorporate automatic data download in docker-compose**

Currently, we have the get_dataset.sh bash script that needs to be run in order to get the data files before deploying locally. This script should be run as a part of docker-compose before the app is run, and that way, the deployed version can use this codebase without git having any of our data files. 

2. (Tech, user study, annotator experience) **Better preprocessing**

Broadly, the current paradigm of chunking documents into maximum 100-word passages or break on newlines results in many pieces of texts that are not ideal for labeling. Cleaning needs to get rid of titles, figure and table descriptions, and other pieces of texts which should not be a part of the labeling process. In terms of getting passages, approaches like text tiling and discourse segmentation need to be looked at, so as to automatically get more logical breaking of the long document into text pieces that annotators should see. For the user study with crowdsourced workers, domain experts should be involved in filtering out text pieces that are too technical and other forms of curation. 

3. (Tech, user study, annotator experience) **Topic modeling update instead of retraining from scratch, faster app, scaleability**

Ideally, the topic model should be able to update as more labels or data comes in rather than being retrained from scratch. This will also make the app faster and more scaleable. Multithreading for parallel processing is also an option to make things faster. For labeled LDA, tomotopy (the current library) does not seem to have an update option. One imperfect solution might be to use a different labaled LDA implementation, such as - https://github.com/JoeZJH/Labeled-LDA-Python - which does seem to have an update function. 

4. (Tech, research) **A better topic model for the use case**

Advances in topic modeling literature can be used. For e.g. SCHOLAR is a topic model that can effectively use document metadata such as labels in a joint manner, and could be useful for human-in-the-loop content analysis. This will also enable separation from previous work as done in ALTO and potentially break new grounds with newer models being compared and evaluated in this human-in-the-loop content analysis paradigm. 

5. (Tech, research) **Better active learning**

Advances in active learning literature should similarly be incorporated, neural classifiers and new strategies can be used as long as they have a really quick runtime in terms of updates since a snappy user experience is of prime importance.

6. (Research) **Pilot user study**

IRB approval was obtained, the pilot user study with the tool in different scenarios can help establish that active learning and topic models can help annotators more efficiently annotate documents for which they do not have a prior label set established, i.e., help humans carry out content analysis for large document collections. 