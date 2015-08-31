#!/usr/bin/python
from itertools import cycle


class Pan(object):

    """Docstring for Pan. """
    SEPARATOR = ':::'
    EMPTY = 'xx'
    TRUTH_FILENAME = 'truth.txt'
    TYPE_REGEX = r'type="(?P<type>.*?)"'
    LANG_REGEX = r'lang="(?P<lang>.*?)"'
    ID_LABEL = 'userid'
    TEXTS_LABEL = 'texts'
    TRAIN_LABEL = 'train'
    TEST_LABEL = 'test'
    COLORS = cycle(['LightGreen', 'Orange', 'Purple', 'Brown', 'Blue'])
