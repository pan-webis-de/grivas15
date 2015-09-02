""" accessor for configuration settings of pan """
import yaml
import os.path
import importlib
from collections import OrderedDict
from features import GroupGenerator, FeatureGenerator

# some global strings
FEATURE_FILE_ATTR = 'feature_file'
CONF_FOLDERNAME = 'config'
LANGS_FOLDERNAME = 'languages'
FEATS_FOLDERNAME = 'features'
FILE_SUFFIX = '.yml'

CLASSIFICATION = 'classification'
REGRESSION = 'regression'


class Config(object):

    """Docstring for Config. """

    def __init__(self, lang):
        """ Configuration representation - read from files

        :lang: The language this configuration is for

        """
        self.lang = lang
        # where to find the settings for the languages
        filename = os.path.join(CONF_FOLDERNAME, LANGS_FOLDERNAME)
        filename = os.path.join(filename, lang + FILE_SUFFIX)
        # each predictor contains the settings for predicting that feature
        with open(filename) as conf_file:
            # keep a dict of settings for each predictor
            self._settings = yaml.load(conf_file)
            self.predictors = {predictor: GroupConfig(predictor, **values)
                               for predictor, values
                               in self._settings.items()
                               if 'data_only' not in values.keys()}

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
    def regression_list(self):
        """ list of regression models

        :returns: list - regression models

        """
        return [pred for pred in self.predictors.keys()
                if self.predictors[pred].type == REGRESSION]

    @property
    def classifier_list(self):
        """ Get list of classifiers

        :returns: list - classifier models

        """
        return [pred for pred in self.predictors.keys()
                if self.predictors[pred].type == CLASSIFICATION]

    @property
    def preprocess_map(self):
        """ The preprocessing mapping for each feature group

        :returns: dict - the mapping

        """
        preprocess = {}
        for pred, value in self.predictors.items():
            if self.predictors[pred].type in [CLASSIFICATION, REGRESSION]:
                preprocess.update(value.preprocess_map)
        return preprocess

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
    def labels(self):
        """ Get a dictionary to easily grab the labels """
        return OrderedDict([each[:2] for each in
                           sorted([(pred, self._settings[pred]['result_label'],
                                  self._settings[pred]['result_column'])
                                  for pred in self._settings.keys()],
                            key=lambda x: x[2])])
        pass


class GroupConfig(object):

    """ Configuration representation for each feature group """

    def __init__(self, label, **kwargs):
        self.label = label
        for name, value in kwargs.items():
            if name == FEATURE_FILE_ATTR:
                filename = os.path.join(CONF_FOLDERNAME, FEATS_FOLDERNAME)
                filename = os.path.join(filename, value)
                with open(filename) as feat_file:
                    settings = yaml.load(feat_file)
                    self.groups = {group: [GeneratorConfig(label, **opts)
                                           for label, opts
                                           in values.items()]
                                   if values else []
                                   for group, values
                                   in settings.items()}
            else:
                setattr(self, name, value)

    @property
    def estimator_model(self):
        """ get the estimator model if an estimator exists
        :returns: The estimator model instantiated with
                  the parameters

        """
        pkg = importlib.import_module(self.estimator_pkg)
        model_class = getattr(pkg, self.estimator)
        # if we have params pass them to constructor
        # else call empty constructor
        params = self.estimator_params
        model = model_class(**params) if params else model_class()
        return model

    @property
    def group_generators(self):
        """ Create GroupGenerator objects for feature groups - reinventing
            sklearn FeatureUnion.. !
        :returns: list of GroupGenerator

        """
        # get a dictionary of GroupGenerators for each style
        # each GroupGenerator contains a list of FeatureGenerators
        # each FeatureGenerator has a name, a function and preprocessing stuff
        # PS. I don't really hate all other programmers.
        return [GroupGenerator(
                self.label,
                group,
                [FeatureGenerator(feat.label,
                                  feat.function,
                                  feat.preprocess_label,
                                  feat.preprocess_funcs)
                 for feat in feats]
                )
                if feats else GroupGenerator(self.label, group, {})
                for group, feats in self.groups.items()
                ]

    @property
    def preprocess_map(self):
        """ Set list of functions to run preprocessing for this feature group
        :returns: mapping of preprocess_labels to list of functions

        """
        return {feat.preprocess_label: feat.preprocess_funcs
                for kind, feats in self.groups.items()
                if feats
                for feat in feats}

    def __repr__(self):
        """ human readable printout!
        :returns: str - my class in ascii

        """
        representation = ''
        for key, value in self.__dict__.items():
            representation += '{}: {}\n'.format(key, value)
        return representation


class GeneratorConfig(object):

    """ Config for each feature generator """

    def __init__(self, label, **kwargs):
        self.label = label
        for name, value in kwargs.items():
            # deal with preprocess later
            if name != 'preprocess':
                setattr(self, name, value)
        # set values for preprocess
        if 'preprocess' in kwargs.keys():
            setattr(self, 'preprocess_label', kwargs['preprocess']['label'])
            setattr(self, 'preprocess_funcs', kwargs['preprocess']['pipeline'])
        # set values to default if no preprocess was defined
        else:
            setattr(self, 'preprocess_label', 'texts')
            setattr(self, 'preprocess_funcs', [])

    def __repr__(self):
        """ human readable printout!
        :returns: str - my class in ascii

        """
        representation = ''
        for key, value in self.__dict__.items():
            representation += '{}: {}\n'.format(key, value)
        return representation
