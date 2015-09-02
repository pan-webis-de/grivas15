import os.path
import regex as re
import pandas as pd
import preprocess
from pan import Pan
from config import Config
from collections import OrderedDict


class DatasetLoader(object):

    """ Pan DatasetLoader class.
        may want to add IdentificationDataset later """

    def __init__(self, path):
        """ Initialize self

        :path: path to data folder
        :lang: language to use
        :media: media to use

        """
        self.folder = path


class ProfilingDataset(DatasetLoader):

    """Docstring for ProfilingDataset. """

    def __init__(self, path, label=Pan.TRAIN_LABEL, percent=100):
        """ Load the profiling datset from this folder

        :path: path to the folder which contains the datafiles
        :label: label to add to data instances
                used with Pan.TRAIN_LABEL, Pan.TEST_LABEL
        :percent: percentage of files from folder to use

        """
        assert 0 <= percent <= 100
        assert label in [Pan.TRAIN_LABEL, Pan.TEST_LABEL]
        DatasetLoader.__init__(self, path)
        self.label = label
        self.percent = percent
        self.config = Config(self.lang)
        self.truth_mapping = self.config.truth_mapping
        self.entries = self._read_entries()
        self.preprocess()
        # TODO see where you are going to put this
        # self.discard_empty()
        # self.discard_duplicates()
        labeldict = self.config.labels
        # create a dictionary on AuthorProfile class to
        # to get the labels
        setattr(AuthorProfile, 'labeldict', OrderedDict())
        for feature, label in labeldict.items():
            AuthorProfile.labeldict[feature] = label

    def __getitem__(self, key):
        """ Access entries with [] with this class
        :returns: TODO

        """
        return self.entries[key]

    def __setitem__(self, key, value):
        """ Modify entries with [] with this class
        :returns: nothing - inserts value to element

        """
        self.entries[key] = value

    def __iter__(self):
        """ Return iterable for this class
        :returns: an iterator over the entries

        """
        return iter(self.entries)

    @property
    def lang(self):
        """ language attribute - gets it from the header of one of the samples.
        :returns: the language this dataset is about

        """
        # pick a sample to find language (don't pick the truth file)
        sample = [choose for choose in os.listdir(self.folder)
                  if 'truth' not in choose][0]
        with open(os.path.join(self.folder, sample), 'r') as sample_file:
            first_line = sample_file.readlines()[0]
            lang = [match.group('lang')
                    for match in re.finditer(Pan.LANG_REGEX, first_line)][0]
        return lang.lower()

    @property
    def media(self):
        """ media attribute - gets the type of media this text is
        :returns: media type of text

        """
        sample = [choose for choose in os.listdir(self.folder)
                  if 'truth' not in choose][0]
        with open(os.path.join(self.folder, sample), 'r') as sample_file:
            first_line = sample_file.readlines()[0]
            try:
                media = [match.group('type')
                         for match in
                         re.finditer(Pan.TYPE_REGEX, first_line)][0]
            except IndexError:
                # TODO pan removed source from header for 2015
                # media = 'unspecified'
                media = 'twitter'
        return media

    def _read_entries(self):
        """ populate entries by reading the data
        :returns: nothing, it populates the entries list

        """
        # If percent_train > 0 we expect a truth file
        lines = []
        entries = []
        try:
            with open(os.path.join(self.folder, Pan.TRUTH_FILENAME)) as tf:
                lines = tf.readlines()
        # if it is a test dataset - there is no truth.txt
        # replicate it
        except IOError:
            # suppose all test files are .xml files
            # TODO handle case where truth -> truth.xml -_-
            test_files = [f for f in os.listdir(self.folder)
                          if f.endswith('.xml')]
            for test_file in test_files:
                userid = test_file.split('.')[0]
                # values are empty for test dataset
                values = sum([(userid,), (Pan.EMPTY,)*self.num_attrs], ())
                line = Pan.SEPARATOR.join(values)
                lines.append(line)
        # Method is the same from here on, since we've gotten or created
        # the lines with the values needed
        for index, line in enumerate(lines):
            # populate dataset with labels ( test or train )
            # train counts from start, test from end <- if they sum to 100
            if self.label == Pan.TRAIN_LABEL:
                if index <= (float(len(lines))/100) * self.percent:
                    entries.append(self._new_instance(line, Pan.TRAIN_LABEL))
            elif self.label == Pan.TEST_LABEL:
                if index >= len(lines) - ((float(len(lines))/100) * self.percent):
                    entries.append(self._new_instance(line, Pan.TEST_LABEL))
        return entries

    def _new_instance(self, line, label):
        """ read line, extract attributes and create and add the
            Author Profile instance

        :line: str - line of data separated with Pan.SEPARATOR
        :label: str - the label to add to the instance
                set to Pan.TRAIN_LABEL or Pan.TEST_LABEL
                but could be used differently
        :returns: an AuthorProfiling instance

        """
        values = line.strip('\n').split(Pan.SEPARATOR)
        ap_attrs = {}
        # get basic stats like lang and media
        ap_attrs.update({'lang': self.lang, 'media': self.media})
        for name, column in self.truth_mapping.items():
            ap_attrs[name] = values[column]
        filename = os.path.join(self.folder, ap_attrs[Pan.ID_LABEL])
        xmlfile = filename + '.xml'
        # open file to get dataset - usually texts
        with open(xmlfile, 'r') as xml:
            content = xml.read()
            docs = preprocess.get_docs(content)
            ap_attrs[Pan.TEXTS_LABEL] = docs
            return AuthorProfile(label=label, **ap_attrs)

    def __repr__(self):
        """ Ipython friendly output
        :returns: str

        """
        return """
                  lang : {} \n
                  media : {} \n
                  texts : \n {}"""\
                      .format(
                             self.lang,
                             self.media,
                             '\n'.join(each.__repr__()
                                       for each in self.entries)
                             )

    def write_data(self, folder):
        """ Write data to the xml file as expected in pan

        :folder: The folder to write output to
        :returns: nothing - it creates a file

        """
        for entry in self.entries:
            filename = os.path.join(folder,
                                    entry.lang + '-' + entry.userid) + '.xml'
            with open(filename, 'w') as output:
                output.write(entry.to_xml())

    def discard_empty(self):
        """ Discard text item if empty
        :returns: nothing

        """
        for index, entry in reversed(list(enumerate(self.entries))):
            if not ''.join(entry.texts):
                del self.entries[index]

    def discard_duplicates(self):
        """ Discard cleaned texts that contain quotations
        :returns: nothing

        """
        for index, entry in enumerate(self.entries):
            with_duplicates = entry.cleaned
            for ind_remove, to_remove in enumerate(self.entries):
                if ind_remove != index:
                    with_duplicates -= to_remove.cleaned.copy()
            self.entries[index].cleaned = with_duplicates

    def preprocess(self):
        """ Preprocess each entry with preprocess steps defined in config
        :returns: nothing - modifies entries in place

        """
        data_instances = self.config.preprocess_map
        for data_label, preprocess_functions in data_instances.items():
            print 'Creating preprocess group: %s' % data_label
            print 'with functions:'
            if preprocess_functions:
                print '- %s' % '\n- '.join(preprocess_functions)
                for entry in self.entries:
                    texts = entry.texts[:]  # copy texts
                    for func_name in preprocess_functions:
                        func = getattr(preprocess, func_name)
                        # TODO get rid of this - it needs to go in conf file
                        # hardcoded stuff!
                        if func_name == 'filter_lang':
                            func(texts, self.lang)
                        else:
                            func(texts)  # functions modify lists in place
                    setattr(entry, data_label, texts)
            else:
                print '- no functions defined'

    def set_labels(self, feature, values):
        """ set the labels of the instances to values

        :feature: the feature to set the value for
        :values: the list of values to set the instances to
        :returns: nothing

        """
        for index, entry in enumerate(self.entries):
            setattr(entry, feature, values[index])

    def get_labels(self, feature):
        """ Get labels of entries for this feature

        :feature: the feature to get the labels for
        :returns: a list of str - labels

        """
        if self.label == Pan.TRAIN_LABEL:
            return zip(*self.get_train(feature=feature))[1]
        else:
            return zip(*self.get_test(feature=feature))[1]

    def get_train(self, label='texts', feature='none'):
        """ Get training data

        :clean: whether to bring the clean data or the raw data
        :feature: what feature the labels are for
        :returns: list of train data

        """
        train = []
        for entry in self.entries:
            if entry.label == Pan.TRAIN_LABEL:
                train.append(entry.datafy(label=label, feature=feature))
        return train

    def get_test(self, label='texts', feature='none'):
        """ Get test data

        :clean: whether to bring the clean data or the raw data
        :feature: what feature the labels are for
        :returns: list of test data

        """
        test = []
        for entry in self.entries:
            if entry.label == Pan.TEST_LABEL:
                test.append(entry.datafy(label=label, feature=feature))
        return test

    # visualization stuff
    def distribution(self, feature):
        """ get and plot the distribution of this feature in the dataset

        :returns: a dataframe with the counts of each feature

        """
        labels = self.get_labels(feature)
        df = pd.DataFrame({'count': [0]*len(labels),
                           'labels': labels})
        df = df.groupby('labels').agg({'count': pd.Series.count})
        df.plot(kind='bar', colormap='summer')
        return df

    def textsize(self, label='texts'):
        """ get and plot the text size for all instances in dataset

        :clean: Whether to get the cleaned data or the raw data
        :returns: a dataframe with the counts for each instance

        """
        if self.label == Pan.TRAIN_LABEL:
            texts = self.get_train(label=label)
        else:
            texts = self.get_test(label=label)
        sizes = [len(text[0]) for text in texts]
        df = pd.DataFrame({'length': sizes})
        ax = df.plot(kind='bar', xticks=[], colormap='summer')
        ax.set_ylabel('characters long')
        ax.set_xlabel('users')
        return df


class AuthorProfile(object):

    """Docstring for AuthorProfile. """

    def __init__(self, label=Pan.TRAIN_LABEL, **kwargs):
        """ an author profile - one user

        """
        self.label = label
        for key, value in kwargs.items():
            try:
                # set psych values
                setattr(self, key, float(value))
            except (ValueError, TypeError):
                setattr(self, key, value)
        # always remove html
        preprocess.clean_html(self.texts)

    def __repr__(self):
        """ IPython friendly output
        :returns: str

        """
        # automatically capture all non iterables
        # (we want custom formatting for text list)
        attr_string = '\n'.join(['%s : %s' % (key, value)
                                 for key, value in self.__dict__.items()
                                 if not hasattr(value, '__iter__')])
        # print a snippet
        texts = ('\ntext: %s \n' % (self.get_text(separator='\n'))
                 .encode('utf-8')[:1000] + ' ...')
        rep = '%s\n %s\n' % (attr_string, texts)
        return rep

    def to_xml(self):
        """Create xml as expected in pan

        :returns: An xml string containing the results of the experiment

        """
        header = "<author "
        body = '\t\t'.join(['%s="%s"\n' % (label, getattr(self, key))
                            for key, label in AuthorProfile.labeldict.items()])
        footer = '/>\n'
        # temp fix for weird truth file choice
        body = body.replace('gender="M"', 'gender="male"')
        body = body.replace('gender="F"', 'gender="female"')
        return header + body + footer

    def get_text(self, label='texts', separator=''):
        """ Get text with preprocess label

        :label: The label of the preprocessed set to get texts from
        :separator: The string to use inbetween texts
        :returns: a string containing all the texts joined with separators

        """
        return separator.join(getattr(self, label))

    def get_label(self, feature):
        """TODO: Docstring for get_label.

        :feature: TODO
        :returns: TODO

        """
        return getattr(self, feature)

    def datafy(self, label='texts', feature='none'):
        """Return a tuple of data - training and label if feature is not none

        :feature: the feature we want the label for
        :returns: tuple of data, label

        """
        if feature == 'none':
            return (self.get_text(label, separator='\n'),)
        else:
            return (self.get_text(label, separator='\n'),
                    getattr(self, feature))
