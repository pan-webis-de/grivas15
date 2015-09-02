#!/usr/bin/python
from itertools import cycle


class Pan(object):

    """Some global constants for Pan. """
    # stuff that has to do with the 3colonsv file
    SEPARATOR = ':::'  # separator used in their 3colonsv
    EMPTY = 'xx'       # denotes values not known - truth values in train set
    TRUTH_FILENAME = 'truth.txt'  # 3colonsv file that contains truth labels

    # in pan dataset user texts are in a xml file - which contains a header
    # below are regexs to capture the information in them
    # things seem to be more complex than needed. Information is split over
    # very many files.
    TYPE_REGEX = r'type="(?P<type>.*?)"'  # capture type from header of file
    LANG_REGEX = r'lang="(?P<lang>.*?)"'  # same for lang

    # dataset class instances have below attributes
    ID_LABEL = 'userid'       # save id in an attribute with this name
    TEXTS_LABEL = 'texts'     # save texts in this attribute
    TRAIN_LABEL = 'train'     # this dataset is a training set
    TEST_LABEL = 'test'       # this dataset is a test set
    # used in visualization
    COLORS = cycle(['LightGreen', 'Orange', 'Purple', 'Brown', 'Blue'])
