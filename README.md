# NCSR Demokritos submission to Pan 2015.

Package consists of a python module and scripts for:
- crossvalidating
- training
- testing
models on the pan 2015 dataset.
It should also work for pan 2014 with slight modifications in the config files.

## Installation:

### Dataset:
In order to run the examples you will need to download the corpus for the author profiling task
from the pan website:

http://www.uni-weimar.de/medien/webis/events/pan-15/pan15-web/author-profiling.html

### Requirements:

Install the requirements 

pip install -r requirements.txt

### Module:

You can also install the module if you would like to check it out from ipython.
git clone this project
cd projectfolder
pip install --user .

## Example usage:

> python train.py -i pan15-author-profiling-training-dataset-2015-04-23/pan15-author-profiling-training-dataset-english-2015-04-23/ -o models


> python test.py -i pan15-author-profiling-training-dataset-2015-04-23/pan15-author-profiling-training-dataset-english-2015-04-23/ -m models/en.bin -o results

> python cross.py -i ../Baseline/pan15-author-profiling-training-dataset-2015-04-23/pan15-author-profiling-training-dataset-english-2015-04-23/

### Output:

> Loading dataset...
> Creating preprocess group: texts
> with functions:
> - no functions defined
> Creating preprocess group: basic
> with functions:
> - clean_html
> - detwittify
> - remove_duplicates
> Loaded 152 users...


--------------- Thy time of Running ---------------

Creating model for en - gender
Using 4 fold validation
Using classifier SVC
beginning cross validation...
Training stylometry , using :
- tfidf ngrams
instances : 114 - features : 20662
Testing stylometry , using :
 -tfidf ngrams
instances : 38 - features : 20662
Training stylometry , using :
- tfidf ngrams
instances : 114 - features : 21043
Testing stylometry , using :
 -tfidf ngrams
instances : 38 - features : 21043
Training stylometry , using :
- tfidf ngrams
instances : 114 - features : 19999
Testing stylometry , using :
 -tfidf ngrams
instances : 38 - features : 19999
Training stylometry , using :
- tfidf ngrams
instances : 114 - features : 20775
Testing stylometry , using :
 -tfidf ngrams
instances : 38 - features : 20775

## Configuration:
