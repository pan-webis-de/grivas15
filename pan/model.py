import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn import cross_validation
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from pan import Pan


class AuthorProfilingModel(object):

    """Docstring for AuthorProfilingModel. """

    def __init__(self, **kwargs):
        """TODO: to be defined1.

        :dataset: AuthorProfiling instance - the dataset to create a model for
        :feature: the feature to build model for, ex: age

        """
        for parameter, value in kwargs.items():
            setattr(self, parameter, value)
        self.lang = self.dataset.lang
        self.group_generators = \
            self.dataset.config[self.feature].group_generators
        # TODO move below to config
        # self.scaler = StandardScaler(with_mean=False)
        self.scaler = StandardScaler()
        self.normalizer = Normalizer()

    def get_params(self, deep=True):
        """ return parameters

        :deep: whether to deep copy
        :returns: dict of parameters

        """
        return {'clf': self.clf,
                'dataset': self.dataset,
                'feature': self.feature
                }

    def set_params(self, **parameters):
        """ set parameters

        :parameters: parameters to set for evaluator - model
        :returns: self

        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def train(self, classifier):
        """train a classifier on the data

        :classifier: the classifier of sklearn to use
        :returns: nothing

        """
        gens = [gen for gen in self.group_generators
                if gen.generators]
        feat_mats, y_mats = zip(*[self.get_data(self.dataset,
                                                gen,
                                                label='train')
                                  for gen in gens])
        X = np.hstack(feat_mats)
        y = y_mats[0]
        self.fit(X, y)
        self.model = classifier

    def test(self, data):
        """ test the trained classifier on test data

        :data: ProfilingDataset - the dataset to test on
        :returns: list of predicted labels for the data instances

        """
        gens = [gen for gen in self.group_generators
                if gen.generators]
        feat_mats, y_mats = zip(*[self.get_data(data, gen, label='test')
                                  for gen in gens])
        X = np.hstack(feat_mats)
        predicted = self.predict(X)
        return predicted

    def cross(self, clf, folds=5, stratified=True):
        """crossvalidate dataset

        :clf: the classifier to use
        :folds: int - k folds
        :returns: list of predicted values

        """
        gens = [gen for gen in self.group_generators
                if gen.generators]
        feat_mats, y_mats = zip(*[self.get_data(self.dataset,
                                                gen,
                                                label='train')
                                  for gen in gens])
        X = np.hstack(feat_mats)
        y = y_mats[0]
        if stratified:
            skf = cross_validation.StratifiedKFold(y, n_folds=folds,
                                                   random_state=13,
                                                   shuffle=True)
        else:
            skf = cross_validation.KFold(len(y), n_folds=folds,
                                         random_state=13, shuffle=True)
        print 'beginning cross validation...'
        scores = cross_validation.cross_val_predict(self, X, y=y, cv=skf)
        # scores = cross_validation.cross_val_score(self, X, y=y, cv=skf,
        #                                           scoring='accuracy')
        return scores

    def decompose(self, func='pca'):
        """ get decomposed features

        :func: the function to use for the decomposition
        :returns: tuple (decomposed data, truth labels)

        """
        gens = [gen for gen in self.group_generators
                if gen.generators]
        feat_mats, y_mats = zip(*[self.get_data(self.dataset,
                                                gen,
                                                label='train')
                                  for gen in gens])
        X = np.hstack(feat_mats).transpose()
        y = y_mats[0]
        gens = [gen for gen in self.group_generators
                if gen.generators]
        features = np.hstack([gen.train(list(matrix))
                              for matrix, gen in
                              zip(np.hsplit(X, len(gens)), gens)])
        decomp = PCA(n_components=2)
        dec_data = decomp.fit(features.toarray()).transform(features.toarray())
        return dec_data, y

    def get_data(self, data, gen, label=Pan.TRAIN_LABEL):
        """ get feature matrix for one generator
        :data: ProfilingDataset - the data to use
        :gen: FeatureGenerator - a FeatureGenerator that produces a matrix,
              get it from the dictionary
        :label: tuple ( train or test, true labels )

        returns: a sparse matrix of extracted features
        """
        assert label in [Pan.TRAIN_LABEL, Pan.TEST_LABEL]
        all_x = []
        for fgen in gen.generators:
            if label == Pan.TRAIN_LABEL:
                x, y = zip(*data.get_train(label=fgen.preprocess_label,
                                           feature=self.feature))
            else:
                x, y = zip(*data.get_test(label=fgen.preprocess_label,
                                          feature=self.feature))
            all_x.append(x)
        # x_feat = gen.test(x)
        # print x_feat.shape
        # return x_feat, y
        X = np.vstack(all_x)
        return X.transpose(), y

    def visualize(self, decomp='pca'):
        """ visualize this dataset using decomp function

        :decomp: function for value decomposition - ex pda, lda
        :returns: nothing - it plots the dataset

        """
        decomp_data, labels = self.decompose(func=decomp)
        df = pd.DataFrame(decomp_data, index=labels)
        ax = None
        for label in set(labels):
            ax = df.ix[label].plot(kind='scatter', x=0, y=1,
                                   color=Pan.COLORS.next(), label=label, ax=ax)

    def fit(self, X, y):
        """ fit method, so that we can pass this class as a
        classifier in sklearn - needed in kfold, because bag
        of words should be from each fold and not from the
        whole set.

        :y: truth values
        :X: the training set
        :returns: TODO

        """
        gens = [gen for gen in self.group_generators
                if gen.generators]
        num_features = sum(len(gen.generators) for gen in gens)
        all_features = []
        start = 0
        for gen in gens:
            gen_feats = len(gen.generators)
            gen_data = np.hsplit(X, num_features)[start: start + gen_feats]
            all_features.append(gen.train(gen_data, y))
            start += gen_feats

        features = sp.hstack(all_features)
        features = self.scaler.fit_transform(features.toarray())
        features = self.normalizer.fit_transform(features)
        print 'instances : {} - features : {}'.format(*features.shape)
        self.clf.fit(features, y)

    def predict(self, X):
        gens = [gen for gen in self.group_generators
                if gen.generators]
        num_features = sum(len(gen.generators) for gen in gens)
        all_features = []
        start = 0
        for gen in gens:
            gen_feats = len(gen.generators)
            gen_data = np.hsplit(X, num_features)[start: start + gen_feats]
            all_features.append(gen.test(gen_data))
            start += gen_feats

        features = sp.hstack(all_features)
        features = self.scaler.transform(features.toarray())
        features = self.normalizer.transform(features)
        print 'instances : {} - features : {}'.format(*features.shape)
        return self.clf.predict(features)
