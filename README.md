# NCSR Demokritos submission to Pan 2015.
##### Pangram, interestingly is actually a word, and means a sentence that contains all the letters of the alphabet.  However, it was chosen to be the name of this project as an anagram of ap-ngram (author profiling with ngrams)

Package consists of a python module and scripts for:
- crossvalidating
- training
- testing
models on the PAN 2015 dataset.
It also works on the PAN 2014 dataset with a slight modification in the config file. For more information
check the [Pan 2014 configuration](#pan-2014-configuration) section.

## Installation:

### Dataset:
In order to run the examples you will need to download the corpus for the author profiling task
from the PAN website:

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

- python train.py -i pan15-author-profiling-training-dataset-2015-04-23/pan15-author-profiling-training-dataset-english-2015-04-23/ -o models

- python test.py -i pan15-author-profiling-training-dataset-2015-04-23/pan15-author-profiling-training-dataset-english-2015-04-23/ -m models/en.bin -o results

- python cross.py -i pan15-author-profiling-training-dataset-2015-04-23/pan15-author-profiling-training-dataset-english-2015-04-23/

#### Output example:

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


> --------------- Thy time of Running ---------------
> 
> Creating model for en - gender
> Using 4 fold validation
> Using classifier SVC
> beginning cross validation...
> Training stylometry , using :
> - tfidf ngrams
> instances : 114 - features : 20662
> Testing stylometry , using :
>  -tfidf ngrams
> instances : 38 - features : 20662
> Training stylometry , using :
> - tfidf ngrams
> instances : 114 - features : 21043
> Testing stylometry , using :
>  -tfidf ngrams
> instances : 38 - features : 21043
> Training stylometry , using :
> - tfidf ngrams
> instances : 114 - features : 19999
> Testing stylometry , using :
>  -tfidf ngrams
> instances : 38 - features : 19999
> Training stylometry , using :
> - tfidf ngrams
> (more info until results)

> Results for en - gender with classifier SVC
> Accuracy mean : 0.796052631579
> Confusion matrix:
>  [[62 14]
>  [17 59]]
> 
> 
> Results for en - age with classifier LinearSVC
> Accuracy mean : 0.736842105263
> Confusion matrix:
>  [[55  3  0  0]
>  [ 7 53  0  0]
>  [ 1 20  1  0]
>  [ 2  7  0  3]]
> 
> 
> Results for en - extroverted with classifier Ridge
> root mean squared error : 0.16351090744
> 
> Results for en - concientious with classifier Ridge
> root mean squared error : 0.161314013176
> 
> Results for en - stable with classifier Ridge
> root mean squared error : 0.224479962212
> 
> Results for en - agreeable with classifier Ridge
> root mean squared error : 0.163949582583
> 
> Results for en - open with classifier Ridge
> root mean squared error : 0.145654367069

## Configuration:
In the config folder is a toy setup of the configuration for pangram. It is based on the
[YAML](http://yaml.org) format.

Settings currently configurable are:
- Pan dataset settings for each language
- Feature groupings, preprocessing for each feature group, and classifier settings

In config/languages there is a file for each language which specifies where each attribute
to be predicted is in the truth file that contains the label for the training set. For each
of these attributes, you can set a file that contains the feature grouping and preprocessing
settings. In the example provided the mapping is the same for each language, but this need
not be the case.

In config/features the settings for each feature group can be found. The format is in the form
label of:
> label of feature group
>  - feature extractor 1
>  - feature extractor 2
>  - ..
>  preprocessing :
>    label: label this so that it doesn't get computed twice if it has been defined elsewhere
>    pipe: 
>        - method 1
>        - method 2
>        - ...
In the above snippet, feature extractor names are expected to be defined in pan/features.py.
Similarly, the above methods are expected to be defined in pan/preprocess.py and process a mutable iterable in place. (in our case a list of texts)

#### Pan 2014 configuration
If you want to try pangram on PAN 2014 you only need to comment out the lines corresponding to the psychometric attributes in config/languages/mylang.yml. Namely, comment out below the age settings - since the other labels didn't exist in Pan 2014.

## License
Pangram - NCSR Demokritos submission for Pan 2015
Copyright 2015 Andreas Grivas

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=======
# pangram
Author Profiling module
