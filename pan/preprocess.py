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
    :returns: list of texts cleaned from html

    """
    output = []
    for text in texts:
        output.append(strip_tags(text))
    return output


def detwittify(texts):
    """ remove characteristics which aren't "natural" text
        hashtags, urls and usernames

    :texts: collection - the iterable of collection to change
    :returns: list of texts cleaned from twitter specific text

    """
    output = []
    for text in texts:
        # remove urls from text
        cleaned = URLREGEX.sub('', text)
        # replace hashtags with the text of the tag
        cleaned = HASHREGEX.sub('\g<2>', cleaned)
        # replace @ tags
        cleaned = REPLYREGEX.sub('', cleaned)
        output.append(cleaned.strip())
    return output


def filter_lang(texts, lang):
    """ Keep only texts identified as written in lang

    :texts: A list of texts to process
    :lang: The language we want to retain texts for
    :returns: list of texts classified as written in lang

    """
    lang_texts = []
    for text in texts:
        if len(text) > 3:
            blob = TextBlob(text)
            if blob.detect_language() == lang:
                lang_texts.append(text)
    return lang_texts


def collapse_repetitions(texts):
    """ Chop longer repetitions of characters to two
        example: Hiiiii -> Hii
        The rationale behind this is that people using different
        lengths of repetition will be captured in the same way
        (less features)
    """
    collapsed = []
    for text in texts:
        collapsed_text = re.sub(r'([\w%s])\1+' % string.punctuation,
                                r'\1\1',
                                text)
        collapsed.append(collapsed_text)
    return collapsed


def remove_duplicates(texts):
    """Remove all duplicates plus original if found
    :returns: list of texts without repetitions

    """
    cleaned = []
    c = Counter(texts)
    for text in texts:
        if not c[text] > 1:
            cleaned.append(text)
    return cleaned


def silhouette(texts):
    """Replace uppercase letters with U
       Replace lowercase letters with l
       Replace repetitions of characters with r
       Replace digits with d
    :returns: list of texts converted to an outline form

    """
    silhouetted = []
    for text in texts:
        sil_text = re.sub(ur'([\p{Lu}])(\1+)',
                          lambda match: chr(31)*(len(match.group(2)) + 1),
                          text)
        sil_text = re.sub(ur'([\p{Ll}])(\1+)',
                          lambda match: chr(30)*(len(match.group(2)) + 1),
                          sil_text)
        sil_text = re.sub(ur'(\p{Lu}+?)', 'U', sil_text)
        sil_text = re.sub(ur'(\p{Ll}+?)', 'l', sil_text)
        sil_text = re.sub(ur'(\d+?)', lambda match: 'd', sil_text)
        silhouetted.append(sil_text)
    return silhouetted
