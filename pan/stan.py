""" Wrapper for Stanford using jnius instead of command line interface """
import os
import jnius


class NER(object):

    """Docstring for NER. """

    def __init__(self, jar, model):
        """Initialize the stan NER module

        :jar: TODO
        :models: TODO

        """
        self._jar = jar
        self._model = model
        try:
            os.environ['CLASSPATH'] = os.environ['CLASSPATH'] +\
                                      os.path.pathsep + jar
        except KeyError:
            os.environ['CLASSPATH'] = jar
        reload(jnius)
        self._crf = jnius.autoclass('edu.stanford.nlp.ie.crf.CRFClassifier')
        self._classifier = self._crf.getClassifier(model)

    def tag(self, text):
        """tag text - returns a list of tokens for each sentence
           that was found in the text according to the stanford module

        :text: TODO
        :returns: TODO

        """
        sents = self._classifier.classify(text).toArray()
        return [[token.toShortString("Answer") for token in sent.toArray()]
                for sent in sents]

    def entity_spans(self, text):
        """tag text - returns a list of tokens for each sentence
           that was found in the text according to the stanford module

        :text: TODO
        :returns: TODO

        """
        sents = self._classifier.classify(text).toArray()
        return sum([[(token.beginPosition(), token.endPosition(),
                      token.toShortString("Answer"))
                     for token in sent.toArray()
                     if token.toShortString("Answer") != 'O']
                    for sent in sents], [])
