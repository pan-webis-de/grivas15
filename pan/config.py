""" accessor for configuration settings of pan """
import yaml
import os.path
from collections import OrderedDict

# some global strings
FEATURE_FILE_ATTR = 'feature_file'
CONF_FOLDERNAME = 'config'
LANGS_FOLDERNAME = 'languages'
RECIPES_FOLDERNAME = 'recipes'
FILE_SUFFIX = '.yml'

CLASSIFICATION = 'classification'
REGRESSION = 'regression'
ONLY_DATA = 'data_only'


class Config(object):

    """ Docstring for Config. """

    def __init__(self, lang):
        """ Configuration representation - read from files

        :lang: The language this configuration is for

        """
        self.lang = lang
        # where to find the settings for the languages
        filename = os.path.join(CONF_FOLDERNAME, LANGS_FOLDERNAME)
        filename = os.path.join(filename, lang + FILE_SUFFIX)
        # where to find the recipes
        recipe_folder = os.path.join(CONF_FOLDERNAME, RECIPES_FOLDERNAME)

        # each predictor contains the settings for predicting that feature
        with open(filename) as conf_file:
            # keep a dict of settings for each predictor
            self._settings = yaml.load(conf_file)
            self.recipes = {label: os.path.join(recipe_folder,
                                                values['recipe'])
                            for label, values
                            in self._settings.items()
                            if ONLY_DATA not in values.keys()}

    def __getitem__(self, key):
        """ Accessor for predictors using key

        :key: The predictor name
        :returns: GroupConfig - the configuration for that predictor

        """
        return self.predictors[key]

    def __setitem__(self, key, value):
        """ Accessor for predictors using key

        :key: The predictor name
        :value: The value to set the key to
        :returns: Nothing

        """
        self.predictors[key] = value

    def __repr__(self):
        """ human readable printout!
        :returns: str - my class in ascii

        """
        representation = ''
        for key, value in self.__dict__.items():
            representation += '{}: {}\n'.format(key, value)
        return representation

    @property
    def truth_mapping(self):
        """ Return mapping of name of attribute to
            corresponding column of truth.txt
        :returns: dict - attr name to int column

        """
        return {pred: self._settings[pred]['column']
                for pred in self._settings.keys()
                if 'column' in self._settings[pred].keys()}

    @property
    def truth_list(self):
        """ Return names of attributes we are trying to classify
            for the task - given in the order they appear in truth.txt
            - this is based on column number in yaml files
        :returns: list of attribute names in order of truth.txt file

        """
        return zip(*
                   sorted(
                      [(pred, self.predictors[pred].column)
                       for pred in self.predictors.keys()],
                      key=lambda x: x[1])
                   )[0]

    @property
    def tasks(self):
        """ Return the names of the tasks we are working on. Tasks are the
            entries that are not data only.
        :returns: list of task names from truth.txt file

        """
        return [label for label, values
                in self._settings.items()
                if ONLY_DATA not in values.keys()]

    @property
    def labels(self):
        """ Get a dictionary to easily grab the labels """
        return OrderedDict([each[:2] for each in
                           sorted([(pred, self._settings[pred]['result_label'],
                                  self._settings[pred]['result_column'])
                                  for pred in self._settings.keys()],
                            key=lambda x: x[2])])

    @property
    def regression_list(self):
        """ list of regression models
        :returns: list - labels of regression models
        """
        return [label for label, values
                in self._settings.items()
                if ONLY_DATA not in values.keys() and
                values['type'] == REGRESSION]

    @property
    def classifier_list(self):
        """ Get list of classifiers
        :returns: list - labels of classifier models
        """
        return [label for label, values
                in self._settings.items()
                if ONLY_DATA not in values.keys() and
                values['type'] == CLASSIFICATION]
