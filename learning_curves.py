#!/usr/bin/python

import math
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from pan import ProfilingDataset, AuthorProfilingModel
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.learning_curve import learning_curve


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


log = []

def cross_validate(dataset, feature, clf, num_folds=4):
    """ train and cross validate a model

    :lang: the language
    :feature: the feature we want to classify for , ex: age

    """

    print '\nCreating model for {} - {}'.format(dataset.lang, feature)
    print 'Using {} fold validation'.format(num_folds)
    print 'Using classifier {}'.format(clf.__class__.__name__)
    # get data
    log.append('\nResults for {} - {} with classifier {}'
               .format(dataset.lang, feature, clf.__class__.__name__))
    model = AuthorProfilingModel(dataset=dataset, feature=feature, clf=clf)
    # scores = model.cross(clf, folds=5)
    # print 'Accuracy scores : {}'.format(scores)
    # print 'Accuracy mean : {}'.format(scores.mean())
    # print 'Accuracy std : {}'.format(scores.std())
    if feature in dataset.config.classifier_list:
        predict = model.cross(clf, folds=num_folds)
        # scores = model.cross(clf, folds=num_folds)
        Y = dataset.get_labels(model.feature)
        # if it's classification we measure micro and macro scores
        # f1_macro = f1_score(Y, predict, average='macro', pos_label=None)
        # f1_micro = f1_score(Y, predict, average='micro', pos_label=None)
        # prec_macro = precision_score(Y, predict, average='macro', pos_label=None)
        # prec_micro = precision_score(Y, predict, average='micro', pos_label=None)
        # rec_macro = recall_score(Y, predict, average='macro', pos_label=None)
        # rec_micro = recall_score(Y, predict, average='micro', pos_label=None)
        accuracy = accuracy_score(Y, predict)
        # accuracy = scores.mean()
        # accuracy_std = scores.std()
        conf = confusion_matrix(Y, predict)
        # log.append('\nF1 macro score : {}'.format(f1_macro))
        # log.append('F1 micro score : {}\n'.format(f1_micro))
        # log.append('Precision macro score : {}'.format(prec_macro))
        # log.append('Precision micro score : {}\n'.format(prec_micro))
        # log.append('Recall macro score : {}'.format(rec_macro))
        # log.append('Recall micro score : {}\n'.format(rec_micro))
        log.append('Accuracy mean : {}'.format(accuracy))
        # log.append('Accuracy std : {}'.format(accuracy_std))
        log.append('Confusion matrix:\n {}\n'.format(conf))
    else:
        predict = model.cross(clf, folds=num_folds, stratified=False)
        Y = dataset.get_labels(model.feature)
        # if it's not, we measure mean square root error (regression)
        sqe = mean_squared_error(Y, predict)
        log.append('root mean squared error : {}'.format(math.sqrt(sqe)))

if __name__ == '__main__':
    parser = ArgumentParser(description='Train a model with crossvalidation'
                            ' on pan dataset - used for testing purposes ')
    parser.add_argument('-i', '--input', type=str,
                        required=True, dest='infolder',
                        help='path to folder with pan dataset for a language')
    parser.add_argument('-f', '--feature', type=str,
                        dest='feature', default='gender',
                        help='feature to plot learning curves for')

    args = parser.parse_args()
    infolder = args.infolder
    feature = args.feature

    print 'Loading dataset...'
    data = ProfilingDataset(infolder, label='train')
    print 'Loaded {} users...\n'.format(len(data.entries))
    config = data.config
    features = config.classifier_list + config.regression_list
    if feature in features:
        clf = config[feature].estimator_model
        X = data.get_train(feature)
        Y = data.get_labels(feature)
        title = "Learning Curves (Naive Bayes)"
        # Cross validation with 100 iterations to get smoother mean test and train
        # score curves, each time with 20% data randomly selected as a validation set.
        cv = cross_validation.ShuffleSplit(digits.data.shape[0], n_iter=100,
                                           test_size=0.2, random_state=0)

        estimator = GaussianNB()
        plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

        title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
        # SVC is more expensive so we do a lower number of CV iterations:
        cv = cross_validation.ShuffleSplit(digits.data.shape[0], n_iter=10,
                                           test_size=0.2, random_state=0)
        plot_learning_curve(estimator, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4)

        plt.show()
    else:
        print('feature "%s" does not exist - try one of the'
              ' following: %s' % (feature, features))
