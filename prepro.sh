#! /bin/bash

# preprocess squad
python -m squad.prepro
python -m squad.convert2class
python -m squad.prepro-class

# preprocess wikiqa
python -m wikiqa.prepro-class

# preprocess semeval
python -m semeval.prepro


