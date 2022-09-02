## This directory contains two notebooks that helped perform power analysis for the user study

1. get_alto_results.ipynb - fetches results from a previous user study that closely mirrors the aims of our own user study, we call that work ALTO in our references here, and the paper is: 

`Poursabzi-Sangdeh, Forough, Jordan Boyd-Graber, Leah Findlater, and Kevin Seppi. "Alto: Active learning with topic overviews for speeding label induction and document labeling." In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1158-1169. 2016.`

This notebook gives us the values we can use to simulate our power analysis in - 

2. power_analysis_final.ipynb - this actually conducts our rough power analysis to find the minimum number of annotators we need to conduct a well-powered study, using ALTO's values to simulate the difference in scores we need between two scenarios being compared in our user study. 