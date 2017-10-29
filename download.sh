#!/usr/bin/env bash

DATA_DIR=$HOME/data

# Download SQuAD
SQUAD_DIR=$DATA_DIR/squad
#mkdir $SQUAD_DIR
#wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -O $SQUAD_DIR/train-v1.1.json
#wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O $SQUAD_DIR/dev-v1.1.json

# Download WikiQA
#wget https://download.microsoft.com/download/E/5/F/E5FCFCEE-7005-4814-853D-DAA7C66507E0/WikiQACorpus.zip -O $DATA_DIR/WikiQACorpus.zip
#unzip $DATA_DIR/WikiQACorpus.zip -d $DATA_DIR
#rm $DATA_DIR/WikiQACorpus.zip

# Download SemEval
SEMEVAL_DIR=$DATA_DIR/semeval
#mkdir $SEMEVAL_DIR
wget http://alt.qcri.org/semeval2016/task3/data/uploads/semeval2016-task3-cqa-ql-traindev-v3.2.zip -O $DATA_DIR/semeval.zip
unzip $DATA_DIR/semeval.zip -d $SEMEVAL_DIR
rm $DATA_DIR/semeval.zip

# Download GloVe
#GLOVE_DIR=$DATA_DIR/glove
#mkdir $GLOVE_DIR
#wget http://nlp.stanford.edu/data/glove.6B.zip -O $GLOVE_DIR/glove.6B.zip
#unzip $GLOVE_DIR/glove.6B.zip -d $GLOVE_DIR

# Download NLTK (for tokenizer)
# Make sure that nltk is installed!
#python -m nltk.downloader -d $HOME/nltk_data punkt
