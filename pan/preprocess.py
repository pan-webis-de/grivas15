import os.path
import regex as re
import string
from textblob import TextBlob
from BeautifulSoup import BeautifulSoup
from HTMLParser import HTMLParser
from stan import NER
from collections import Counter

# globals
textregex = re.compile(r'!\[CDATA\[(?P<text>.*?)\]', re.DOTALL | re.UNICODE)
urlregex = re.compile(r'(?P<all>\s?(?P<url>(https?|ftp)://[^\s/$.?#].[^\s]*))')
# urlregex.sub('', text)
hashregex = re.compile(r'(?<=\s+|^)(?P<all>#(?P<content>\w+))', re.UNICODE)
# hashregex.sub('\g<2>', 'this #hashtag is cool #imo')
replyregex = re.compile(r'(?P<all>(^|\s*)@(?P<content>\w+)\s*?)', re.UNICODE)
# replyregex.sub('\g<2>', 'this #hashtag is cool #imo')
viaregex = re.compile(r'(via *((@\w+\s?)|(https?://[^\s/$.?#].[^\s]*)))',
                      re.UNICODE)
saidregex = re.compile(r'((^| *|") *@\w+ *:)', re.UNICODE)
quoteregex = re.compile(r'((?P<quote>"?.*"?) *[:\n\-.](?P<namedentity>.*))',
                        re.UNICODE)

# utility methods

stanford_jar = os.path.abspath('resources/stanford-ner-2014-06-16/'
                               'stanford-ner-3.4.jar')
stanford_model = os.path.abspath('resources/stanford-ner-2014-06-16/'
                                 'classifiers/english.all.3class.distsim'
                                 '.crf.ser.gz')
ner = None


def quote_found(text):
    """check if current text is a quotation - use ner

    :text: TODO
    :returns: TODO

    """
    global ner
    if not ner:
        ner = NER(stanford_jar, stanford_model)
    for match in quoteregex.finditer(text):
        candidate = match.group('namedentity')
        sent_ents = ner.tag(candidate)
        for ents in sent_ents:
            if 'PERSON' in ents:
                return True
    return False

# TODO add quote - NER
# add other quote styles
quoteregexs = [viaregex.findall,
               saidregex.findall,
               quote_found
               ]


def get_docs(text):
    """ Return documents from pan xml style data
        documents are in <document> tags

    :text: The xml as text
    :returns: list - list of str - documents

    """
    bs = BeautifulSoup(text)
    # remove right tabs - why do you add tabs in CDATA?
    docs = [doc.text.rstrip('\t') for doc in bs.findAll('document')]
    return docs


def strip_tags(html):
    """ Strip html tags from text

    :text: The text with html to be removed
    :returns: str - the text
    """
    class MLStripper(HTMLParser):
        def __init__(self):
            self.reset()
            self.fed = []

        def handle_data(self, d):
            self.fed.append(d)

        def get_data(self):
            return ''.join(self.fed)
    s = MLStripper()
    s.feed(html)
    return s.get_data()


def clean_html(texts):
    """ Strip html

    :texts: collection - the collection of texts to change
    :returns: nothing - modifies collection in place

    """
    for index, text in reversed(list(enumerate(texts))):
        texts[index] = strip_tags(text)
        # remove if empty string
        if not texts[index].strip():
            del texts[index]


def detwittify(texts):
    """ remove characteristics which aren't author dependent like hashes and urls

    :texts: collection - the iterable of collection to change
    :returns: nothing - modifies collection in place

    """
    for index, each in enumerate(texts):
        # remove urls from text
        texts[index] = urlregex.sub('', each)
        # replace hashtags with the text of the tag
        texts[index] = hashregex.sub('\g<2>', texts[index])
        # replace @ tags
        texts[index] = replyregex.sub('', texts[index])
        texts[index] = texts[index].strip()


def clear_quotations(texts):
    """ Remove tweets that contain quotations
    :returns: number - a count of how many quotations where removed

    """
    for index, text in reversed(list(enumerate(texts))):
        for func in quoteregexs:
            matches = func(text)
            if matches:
                del texts[index]
                break


def filter_lang(texts, lang):
    """ Keep only texts identified as written in lang

    :texts: A list of texts to process
    :lang: The language we want to retain texts for
    :returns: Nothing, modifies list in place retaining only texts
              identified as written in the language we want

    """
    for index, text in reversed(list(enumerate(texts))):
        if len(text) > 3:
            blob = TextBlob(text)
            if not blob.detect_language() == lang:
                del texts[index]


def collapse_repetitions(texts):
    """ Replace reptitions with 2 chars
    :returns: TODO

    """
    for index, each in enumerate(texts):
        # remove urls from text
        texts[index] = re.sub(r'([\w%s])\1+' % string.punctuation,
                              r'\1\1',
                              texts[index])


def collapse_entities(texts):
    """ collapse entities to a single character
    :returns: TODO

    """
    global ner
    if not ner:
        ner = NER(stanford_jar, stanford_model)
    for index, text in enumerate(texts):
        ent_spans = ner.entity_spans(text)
        # want this to be fast - many joins
        string = ''
        current = 0
        for begin, end, tag in ent_spans:
            string += text[current:begin]
            # string += tag[0]
            current = end
        if current:
            string += text[current:]
            # we found some entities
            texts[index] = string


def remove_duplicates(texts):
    """Remove all duplicates plus original if found
    :returns: Nothing, modifies the list in place

    """
    c = Counter(texts)
    for index, text in reversed(list(enumerate(texts))):
        if c[text] > 1:
            del texts[index]


def silhouette(texts):
    """Replace uppercase letters with U
       Replace lowercase letters with l
       Replace repetitions of characters with r
       Replace digits with d
    :returns: TODO

    """
    for index, each in enumerate(texts):
        # remove urls from text
        texts[index] = re.sub(ur'([\p{Lu}])(\1+)',
                              lambda match: chr(31)*(len(match.group(2)) + 1),
                              texts[index])
        texts[index] = re.sub(ur'([\p{Ll}])(\1+)',
                              lambda match: chr(30)*(len(match.group(2)) + 1),
                              texts[index])
        texts[index] = re.sub(ur'(\p{Lu}+?)',
                              'U',
                              texts[index])
        texts[index] = re.sub(ur'(\p{Ll}+?)',
                              'l',
                              texts[index])
        texts[index] = re.sub(ur'(\d+?)',
                              lambda match: 'd',
                              texts[index])
