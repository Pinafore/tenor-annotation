#!/bin/bash
mkdir -p backend/data
wget -P backend/data/ https://obj.umiacs.umd.edu/nist-data/corpus.pkl
wget -P backend/data/ https://obj.umiacs.umd.edu/nist-data/lda_model.bin
wget -P backend/data/ https://obj.umiacs.umd.edu/nist-data/nist_ras_documents_cleaned.csv
