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
import regex as re
import nltk
import numpy
from textblob.tokenizers import WordTokenizer
from sklearn.base import BaseEstimator, TransformerMixin

# ------------------------ feature generators --------------------------------#


class TopicTopWords(BaseEstimator, TransformerMixin):

    """ Suppose texts can be split into n topics. Represent each text
        as a percentage for each topic."""

    def __init__(self, n_topics, k_top):
        import lda
        from sklearn.feature_extraction.text import CountVectorizer
        self.n_topics = n_topics
        self.k_top = k_top
        self.model = lda.LDA(n_topics=self.n_topics,
                             n_iter=10,
                             random_state=1)
        self.counter = CountVectorizer()

    def fit(self, X, y=None):
        X = self.counter.fit_transform(X)
        self.model.fit(X)
        return self

    def transform(self, texts):
        """ transform data

        :texts: The texts to count hashes in
        :returns: list of counts for each text

        """
        X = self.counter.transform(texts).toarray()  # get counts for each word
        topic_words = self.model.topic_word_  # model.components_ also works
        topics = numpy.hstack([X[:, numpy.argsort(topic_dist)]
                                [:, :-(self.k_top+1):-1]
                              for topic_dist in topic_words])
        return topics


class CountHash(BaseEstimator, TransformerMixin):

    """ Model that extracts a counter of twitter-style hashes. """

    pat = re.compile(r'(?<=\s+|^)#\w+', re.UNICODE)

    def fit(self, X, y=None):
        return self

    def transform(self, texts):
        """ transform data

        :texts: The texts to count hashes in
        :returns: list of counts for each text

        """
        return [[len(CountHash.pat.findall(text))] for text in texts]


class CountReplies(BaseEstimator, TransformerMixin):

    """ Model that extracts a counter of twitter-style @replies. """

    pat = re.compile(r'(?<=\s+|^)@\w+', re.UNICODE)

    def fit(self, X, y=None):
        return self

    def transform(self, texts):
        """ transform data

        :texts: The texts to count replies in
        :returns: list of counts for each text

        """
        return [[len(CountReplies.pat.findall(text))] for text in texts]


class CountURLs(BaseEstimator, TransformerMixin):

    """ Model that extracts a counter of URL links from text. """

    pat = re.compile(r'((https?|ftp)://[^\s/$.?#].[^\s]*)')

    def fit(self, X, y=None):
        return self

    def transform(self, texts):
        """ transform data

        :texts: The texts to count URLs in
        :returns: list of counts for each text

        """
        return [[len(CountURLs.pat.findall(text))] for text in texts]


class CountCaps(BaseEstimator, TransformerMixin):

    """ Model that extracts a counter of capital letters from text. """

    def fit(self, X, y=None):
        return self

    def transform(self, texts):
        """ transform data

        :texts: The texts to count capital letters in
        :returns: list of counts for each text

        """
        return [[sum(c.isupper() for c in text)] for text in texts]


class CountWordCaps(BaseEstimator, TransformerMixin):

    """ Model that extracts a counter of capital words from text. """

    def fit(self, X, y=None):
        return self

    def transform(self, texts):
        """ transform data

        :texts: The texts to count capital words in
        :returns: list of counts for each text

        """
        return [[sum(w.isupper() for w in nltk.word_tokenize(text))]
                for text in texts]


class CountWordLength(BaseEstimator, TransformerMixin):

    """ Model that extracts a counter of word length from text. """

    def __init__(self, span):
        """ Initialize this feature extractor
        :span: tuple - range of lengths to count

        """
        self.span = span

    def fit(self, X, y=None):
        return self

    def transform(self, texts):
        """ transform data

        :texts: The texts to count word lengths in
        :returns: list of counts for each text

        """
        mini, maxi = self.span
        num_counts = maxi - mini
        wt = WordTokenizer()
        tokens = [wt.tokenize(text) for text in texts]
        text_len_dist = []
        for line_tokens in tokens:
            counter = [0]*num_counts
            for word in line_tokens:
                word_len = len(word)
                if mini <= word_len <= maxi:
                    counter[word_len - 1] += 1
            text_len_dist.append([each for each in counter])
        return text_len_dist
