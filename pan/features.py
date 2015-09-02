""" Module containing feature generators used for learning.
    I think I reinvented sklearn pipelines - too late now!
    A dictionary of functions is used for feature generation.
    If a function has only one argument feature generation is
    independent of training or test case.
    If it takes two arguments, feature generation depends
    on case - for example: bag_of_words
    This is supposed to be extensible as you can add or remove
    any functions you like from the dictionary
"""
# I actually did reinvent sklearn FeatureUnion + sklearn Pipeline
# TODO: revert to sklearn equivalents to make code much simpler
import os
import regex as re
import numpy as np
import string
import nltk
import inspect
# from pyinsect import CentroidModel
from textblob import TextBlob
from textblob.tokenizers import WordTokenizer
from scipy.sparse import vstack, hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models.word2vec import Word2Vec

# ------------------------ feature generators --------------------------------#
# ------------------ for heavy weaponry see bottom ---------------------------#


# basically we have 3 types of feature generator functions.
# 1. A function that depends only on the X input when extracting features for
#    either train or test
# 2. A function that depends only on the X input - but needs the fit of the
#    training data in order to fit to test data - for example the vocabulary
#    of tfidf is needed to fit the test data.
#    Therefore in this case we also need the model built on the train data
#    when fitting the test data
# 3. A function that depends on X and also needs the labels in order to create
#    the model. Therefore in this case we need 2 inputs for the train case
#    and 3 inputs for the test case.

# functions with this annotation correspond to group 2
def training_dependent(func):
    """ training dependent - pass X, Y (need Y) """
    def wrapper(train, test, model):
        if train:
            return func(train, test, model)
        else:
            return func(train, test, model)
    return wrapper


# functions with this annotation correspond to group 3
def training_independent(func):
    """ training independent - pass X only """
    def wrapper(train, test, model):
        if train:
            return func(train[0], test, model)
        else:
            return func(train, test, model)
    return wrapper


# all below with one argument correspond to aforementioned group 1
def count_hash(data):
    """Counts number of hash occurences

    :data: the list of texts or a single text to count from
    :returns: the number of hash tags
    :         If called with a list, expect a list

    """
    pat = re.compile(r'(?<=\s+|^)#\w+', re.UNICODE)
    if hasattr(data, '__len__'):
        return vstack([len(pat.findall(each)) for each in data])
    else:
        return len(pat.findall(data))


def count_reply(data):
    """Counts number of reply tag occurences

    :data: the list of texts or a single text to count from
    :returns: the number of reply tags
    :         If called with a list, expect a list

    """
    pat = re.compile(r'(?<=\s+|^)@\w+', re.UNICODE)
    if hasattr(data, '__len__'):
        return vstack([len(pat.findall(each)) for each in data])
    else:
        return len(pat.findall(data))


def count_url_links(data):
    """Counts number of links in text

    :data: the list of texts or a single text to count from
    :returns: the number of url links in the text
    :         If called with a list, expect a list

    """
    pat = re.compile(r'((https?|ftp)://[^\s/$.?#].[^\s]*)')
    if hasattr(data, '__len__'):
        return vstack([len(pat.findall(each)) for each in data])
    else:
        return len(pat.findall(data))


def count_money(data):
    """Counts number of money like strings

    :data: the list of texts or a single text to count from
    :returns: the number of money like strings
    :         If called with a list, expect a list

    """
    pat = re.compile(r'\$\d+[,.]?\d*')
    if hasattr(data, '__len__'):
        return vstack([len(pat.findall(each)) for each in data])
    else:
        return len(pat.findall(data))


def count_caps(data):
    """Counts capital letters in text

    :data: the list of texts or a single text to count from
    :returns: number of capital letters
    :         If called with a list, expect a list

    """
    if hasattr(data, '__len__'):
        return vstack([sum(c.isupper() for c in each) for each in data])
    else:
        return sum(c.isupper() for c in data)


def count_word_caps(data):
    """Counts capital words in text

    :data: the list of texts or a single text to count from
    :returns: number of capital words
    :         If called with a list, expect a list

    """
    if hasattr(data, '__len__'):
        return vstack([sum(w.isupper() for w in nltk.word_tokenize(each))
                       for each in data])
    else:
        return sum(w.isupper() for w in nltk.word_tokenize(data))


def count_word_lower(data):
    """Counts lowercase words in text

    :data: the list of texts or a single text to count from
    :returns: number of lowercase words
    :         If called with a list, expect a list

    """
    if hasattr(data, '__len__'):
        return vstack([sum(w.islower() for w in nltk.word_tokenize(each))
                       for each in data])
    else:
        return sum(w.islower() for w in nltk.word_tokenize(data))


def count_word_title(data):
    """Counts titlecase words in text

    :data: the list of texts or a single text to count from
    :returns: number of titlecase words
    :         If called with a list, expect a list

    """
    if hasattr(data, '__len__'):
        return vstack([sum(w.istitle() for w in nltk.word_tokenize(each))
                       for each in data])
    else:
        return sum(w.istitle() for w in nltk.word_tokenize(data))


def count_punct(data):
    """Counts punctuation in text

    :data: the list of texts or a single text to count from
    :returns: number of punctuation characters
    :         If called with a list, expect a list

    """
    if hasattr(data, '__len__'):
        return vstack([sum(c in string.punctuation for c in each)
                       for each in data])
    else:
        return sum(c in string.punctuation for c in data)


def count_word_length(data):
    """Counts word length distribution in text

    :data: the list of texts or a single text to count from
    :returns: A list of frequencies words with 1-19 letters
    :         If called with a list, expect an array

    """
    wt = WordTokenizer()
    tokens = [wt.tokenize(line) for line in data]
    # tokens = [line.split() for line in data]
    text_len_dist = []
    for line_tokens in tokens:
        counter = [0]*20
        for word in line_tokens:
            word_len = len(word)
            if word_len <= 20:
                counter[word_len - 1] += 1
        text_len_dist.append(hstack([each for each in counter]))
    return vstack(text_len_dist)


def get_polarity(data):
    """ Returns the sentiment polarity of a text

    :data: the list of texts or a single text to count from
    :returns: the sentiment polarity as a number between -1 and 1
    :         If called with a list, expect an array

    """
    if hasattr(data, '__len__'):
        return vstack([hstack(TextBlob(each).sentiment[:]) for each in data])
    else:
        return TextBlob(data).sentiment.polarity


# @training_dependent
# def ngram_graphs(train=None, test=None, model=None):
#     """ Creates centroid graphs for every class
#         using percentage of training.

#     :train: A list of training texts
#     :test: A list of test texts
#     :model: The model learned if called for test data
#     :returns: Distance of each instance from centroids

#     """
#     if test is None:
#         if hasattr(train, '__len__'):
#             model = CentroidModel(percentage=0.8)
#             train_feat = model.fit(train[0], train[1])
#             return {'train': train_feat, 'model': model}
#         else:
#             raise TypeError('data must be a list of texts')
#     # case only test
#     elif train is None:
#         if hasattr(test, '__len__'):
#             test_feat = model.transform(test)
#             return {'test': test_feat}
#         else:
#             raise TypeError('data must be a list of texts')
#     else:
#         raise AttributeError('No specified model or train or test set')


@training_independent
def bag_of_ngrams(train=None, test=None, model=None):
    """ Creates a set of words found in your texts and stores counts
        of each and every one of them

    :train: A list of training texts
    :test: A list of test texts
    :model: The model learned if called for test data
    :returns: An array of counts of the words in the set for each text

    """
    # case only train
    if test is None:
        if hasattr(train, '__len__'):
            vec = CountVectorizer(dtype=float, analyzer='char', ngram_range=(3, 3))
            train_feat = vec.fit_transform(train)
            return {'train': train_feat, 'model': vec}
        else:
            raise TypeError('data must be a list of texts')
    # case only test
    elif train is None:
        if hasattr(test, '__len__'):
            test_feat = model.transform(test)
            return {'test': test_feat}
        else:
            raise TypeError('data must be a list of texts')
    else:
        raise AttributeError('No specified model or train or test set')


@training_independent
def bag_of_words(train=None, test=None, model=None):
    """ Creates a set of words found in your texts and stores counts
        of each and every one of them

    :train: A list of training texts
    :test: A list of test texts
    :returns: An array of counts of the words in the set for each text

    """
    # case only train
    if test is None:
        if hasattr(train, '__len__'):
            vec = CountVectorizer(dtype=float, analyzer='word')
            train_feat = vec.fit_transform(train)
            return {'train': train_feat, 'model': vec}
        else:
            raise TypeError('data must be a list of texts')
    # case only test
    elif train is None:
        if hasattr(test, '__len__'):
            test_feat = model.transform(test)
            return {'test': test_feat}
        else:
            raise TypeError('data must be a list of texts')
    else:
        raise AttributeError('No specified model or train or test set')


@training_independent
def tfidf_related_words(train=None, test=None, model=None):
    """ Creates a set of words found in your texts and stores counts
        of each and every one of them

    :train: A list of training texts
    :test: A list of test texts
    :returns: An array of counts of the words in the set for each text

    """
    # case only train
    if test is None:
        if hasattr(train, '__len__'):
            similar = ['love', 'bye', 'not', 'yes', 'no', 'thank', 'you',
                       'fuck', 'hate', 'it', 'if', 'we', 'brother',
                       'friend', 'yeah', 'tomorrow', 'bitch', 'dad',
                       'really', 'every', 'crap', 'morning',
                       'never', 'cunt', 'noob', 'listen', 'bored']
            modelpath = os.path.abspath('resources/word2vec/vectors.bin')
            w2v = Word2Vec.load_word2vec_format(modelpath, binary=True)
            vocab = {related for word in similar
                     for related, sim in
                     w2v.most_similar(positive=[word], topn=30)}
            vec = TfidfVectorizer(dtype=float, analyzer='word', vocabulary=vocab)
            train_feat = vec.fit_transform(train)
            return {'train': train_feat, 'model': vec}
        else:
            raise TypeError('data must be a list of texts')
    # case only test
    elif train is None:
        if hasattr(test, '__len__'):
            test_feat = model.transform(test)
            return {'test': test_feat}
        else:
            raise TypeError('data must be a list of texts')
    else:
        raise AttributeError('No specified model or train or test set')


@training_independent
def bag_of_smileys(train=None, test=None, model=None):
    """ Get tf-idf features for train and test data

    :train: A list of training texts
    :test: A list of test texts
    :returns: An array of frequencies of the smileys in the set for each text

    """
    smileys = """^^ ^_^ ): (: <3 :) :( :s :d =d :-) :-( ;) xd ;3 :|
                 :3 :'v c: :0 :o :/ lol afk wtf omg"""
    # case given train and test
    if test is None:
        if hasattr(train, '__len__'):
            vec = CountVectorizer(dtype=float, vocabulary=smileys.split())
            train_feat = vec.fit_transform(train)
            return {'train': train_feat, 'model': vec}
        else:
            raise TypeError('data must be a list of texts')
    # case only test
    elif train is None:
        if hasattr(test, '__len__'):
            test_feat = model.transform(test)
            return {'test': test_feat}
        else:
            raise TypeError('data must be a list of texts')
    else:
        raise AttributeError('No specified model or train or test set')


@training_independent
def bag_of_hash(train=None, test=None, model=None):
    """ Get bag of hash features for train and test data

    :train: A list of training texts
    :test: A list of test texts
    :returns: An array of frequencies of the hashtags in the set for each text

    """
    # case only train
    if test is None:
        if hasattr(train, '__len__'):
            vec = CountVectorizer(dtype=float, analyzer='word', token_pattern=r'#\w+')
            train_feat = vec.fit_transform(train)
            return {'train': train_feat, 'model': vec}
        else:
            raise TypeError('data must be a list of texts')
    # case only test
    elif train is None:
        if hasattr(test, '__len__'):
            test_feat = model.transform(test)
            return {'test': test_feat}
        else:
            raise TypeError('data must be a list of texts')
    else:
        raise AttributeError('No specified model or train or test set')


@training_independent
def bag_of_punct(train=None, test=None, model=None):
    """ Creates a set of characters found in your texts and stores counts
        of each and every one of them

    :train: A list of training texts
    :test: A list of test texts
    :returns: An array of counts of the characters in the set for each text

    """
    # case only train
    if test is None:
        if hasattr(train, '__len__'):
            vec = CountVectorizer(dtype=float, analyzer='char')
            train_feat = vec.fit_transform(train)
            return {'train': train_feat, 'model': vec}
        else:
            raise TypeError('data must be a list of texts')
    # case only test
    elif train is None:
        if hasattr(test, '__len__'):
            test_feat = model.transform(test)
            return {'test': test_feat}
        else:
            raise TypeError('data must be a list of texts')
    else:
        raise AttributeError('No specified model or train or test set')


@training_independent
def tfidf_ngrams(train=None, test=None, model=None):
    """ Creates a set of words found in your texts and stores counts
        of each and every one of them

    :train: A list of training texts
    :test: A list of test texts
    :returns: An array of counts of the words in the set for each text

    """
    # case only train
    if test is None:
        if hasattr(train, '__len__'):
            vec = TfidfVectorizer(dtype=float, analyzer='char', ngram_range=(3, 3))
            train_feat = vec.fit_transform(train)
            return {'train': train_feat, 'model': vec}
        else:
            raise TypeError('data must be a list of texts')
    # case only test
    elif train is None:
        if hasattr(test, '__len__'):
            test_feat = model.transform(test)
            return {'test': test_feat}
        else:
            raise TypeError('data must be a list of texts')
    else:
        raise AttributeError('No specified model or train or test set')


@training_independent
def tfidf_words(train=None, test=None, model=None):
    """ Creates a set of words found in your texts and stores counts
        of each and every one of them

    :train: A list of training texts
    :test: A list of test texts
    :returns: An array of counts of the words in the set for each text

    """
    # case only train
    if test is None:
        if hasattr(train, '__len__'):
            vec = TfidfVectorizer(dtype=float, analyzer='word', max_features=1000)
            train_feat = vec.fit_transform(train)
            return {'train': train_feat, 'model': vec}
        else:
            raise TypeError('data must be a list of texts')
    # case only test
    elif train is None:
        if hasattr(test, '__len__'):
            test_feat = model.transform(test)
            return {'test': test_feat}
        else:
            raise TypeError('data must be a list of texts')
    else:
        raise AttributeError('No specified model or train or test set')


def num_args(func):
    """ Get the number of arguments of a function

    :func: The function to inspect the arguments
    :returns: The number of arguments func has

    """
    return len(inspect.getargspec(func)[0])


def featurize(train, test, featuremap, model='None'):
    """Creates rows of features for a given list of data

    :text: the input texts to create features for
    :returns: a list of lists of feature values for the input texts

    """
    assert type(train) == type(test)
    train_data, test_data = zip(*[(func(train), func(test))
                                if num_args(func) == 1
                                else func(train, test, model)
                                for func in featuremap.values()])
    return hstack(train_data), hstack(test_data)


class GroupGenerator(object):

    """Models a machine learning feature grouping generator"""

    def __init__(self, classification_task, group, generators=None):
        """ Initialize a GroupGenerator

        :lang: The language the feature generator is for
        :classification_task: The types of features we have -
                              groups of features
        :group: The group of features this belongs to
        :generators: The mapping of this group to features - FeatureGenerator

        """
        self.group = group
        self.classification_task = classification_task
        self.generators = generators

    def train(self, X, y):
        """ Fit data - equivalent to sklearn fit

        :X: train data
        :y: labels if they exist
        :returns: The result of fitting the data

        """
        outdata = []
        print 'Training {} , using :'.format(self.group)
        # horizontally stack results from each feature group
        # reinventing sklearn feature union.
        for data, generator in zip(X, self.generators):
            outdata.append(generator.train(data, y))
        return hstack(outdata)

    def test(self, X):
        """ Fit test data - equivalent to sklearn transform

        :X: test data to fit
        :returns: The results of fitting the data

        """
        outdata = []
        print 'Testing {} , using :'.format(self.group)
        for data, generator in zip(X, self.generators):
            outdata.append(generator.test(data))
        return hstack(outdata)

    def __repr__(self):
        """ human readable printout!
        :returns: str - my class in ascii

        """
        representation = '\n\n--- group generator ---\n'
        for key, value in self.__dict__.items():
            representation += '{}: {}\n'.format(key, value)
        return representation


class FeatureGenerator(object):

    def __init__(self, name, function, preprocess_label, preprocess_funcs):
        """ Initialize a FeatureGenerator for a feature

        :name: Name of feature generator
        :function: The function that generates this feature
        """

        self.name = name
        self.function = function
        self.preprocess_label = preprocess_label
        self.preprocess_funcs = preprocess_funcs
        self.model = None

    def train(self, X, y):
        """ Trains this model.

        :X: The training set
        :y: The labels
        :returns: Trained model (this is actually equivalent to sklearn fit)

        """
        print '- %s' % self.name
        # make sure this is not 2D
        # collapse it if it is
        X = np.reshape(X, (len(X),))
        func = globals()[self.function]
        # Some feature generators don't need labels
        # in order to generate features - for example counts of words
        # so if the function only accepts one argument, just pass X
        # this is ugly - but i'm not going to change all the functions now
        if num_args(func) == 1:
            return func(X)
        else:
            results = func([X, y], None, None)
            self.model = results['model']
        return results['train']

    def test(self, X):
        """ Evaluate on model

        :X: Input dataset - list of instances
        :returns: The results of evaluating the model -
                  equivalent to sklearn transform.

        """
        print ' -%s' % self.name
        # make sure this is not 2D
        # collapse it if it is
        X = np.reshape(X, (len(X),))
        func = globals()[self.function]
        if num_args(func) == 1:
            return func(X)
        else:
            try:
                results = func(None, X, self.model)
            except KeyError:
                raise AttributeError('no training has been done'
                                     ', use train first')
        return results['test']

    def __repr__(self):
        """ human readable printout!
        :returns: str - my class in ascii

        """
        representation = '\n\n--- feature generator ---\n'
        for key, value in self.__dict__.items():
            representation += '{}: {}\n'.format(key, value)
        return representation
