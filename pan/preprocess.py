import regex as re
import string
from textblob import TextBlob
from BeautifulSoup import BeautifulSoup
from HTMLParser import HTMLParser
from collections import Counter

# regexps to capture hashtags - replies and urls
URLREGEX = re.compile(r'(?P<all>\s?(?P<url>(https?|ftp)://[^\s/$.?#].[^\s]*))')
HASHREGEX = re.compile(r'(?<=\s+|^)(?P<all>#(?P<content>\w+))', re.UNICODE)
REPLYREGEX = re.compile(r'(?P<all>(^|\s*)@(?P<content>\w+)\s*?)', re.UNICODE)


# used to strip html from text
class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return ''.join(self.fed)


# --------------------- helper methods --------------------------

def get_docs(xml):
    """ Return documents from pan xml style data (pan 2014 dataset)
        documents are in <document> tags

    :xml: str - content of file to clean from xml
    :returns: list - list of str - documents

    """
    bs = BeautifulSoup(xml)
    # remove right tabs - why do you add tabs in CDATA?
    docs = [doc.text.rstrip('\t') for doc in bs.findAll('document')]
    return docs


def strip_tags(html):
    """ Strip html tags from text

    :text: The text with html to be removed
    :returns: str - the text
    """
    s = MLStripper()
    s.feed(html)
    return s.get_data()


# --------------------- preprocessing methods --------------------------
# all these methods modify the data in place. They expect a mutable iterable
# containing the texts as an input

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
    """
    for index, each in enumerate(texts):
        # remove urls from text
        texts[index] = URLREGEX.sub('', each)
        # replace hashtags with the text of the tag
        texts[index] = HASHREGEX.sub('\g<2>', texts[index])
        # replace @ tags
        texts[index] = REPLYREGEX.sub('', texts[index])
        texts[index] = texts[index].strip()


def filter_lang(texts, lang):
    """ Keep only texts identified as written in lang

    :texts: A list of texts to process
    :lang: The language we want to retain texts for
    """
    for index, text in reversed(list(enumerate(texts))):
        if len(text) > 3:
            blob = TextBlob(text)
            if not blob.detect_language() == lang:
                del texts[index]


def collapse_repetitions(texts):
    """ Chop longer repetitions of characters to two
        example: Hiiiii -> Hii
        The rationale behind this is that people using different
        lengths of repetition will be captured in the same way
        (less features)
    """
    for index, each in enumerate(texts):
        # remove urls from text
        texts[index] = re.sub(r'([\w%s])\1+' % string.punctuation,
                              r'\1\1',
                              texts[index])


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
    :returns: Text converted to an outline form

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
