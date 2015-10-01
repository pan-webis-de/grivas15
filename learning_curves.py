#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from pan import ProfilingDataset
from sklearn import cross_validation
from sklearn.learning_curve import learning_curve
from sklearn.metrics import precision_score, recall_score, accuracy_score, \
                            f1_score, confusion_matrix, mean_squared_error
from tictacs import from_recipe

# below code taken and adapted from example
# @ http://scikit-learn.org/stable/auto_examples/plot_learning_curve.html
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 6)):
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


if __name__ == '__main__':
    parser = ArgumentParser(description='Train a model with crossvalidation'
                            ' on pan dataset - used for testing purposes ')
    parser.add_argument('-i', '--input', type=str,
                        required=True, dest='infolder',
                        help='path to folder with pan dataset for a language')
    parser.add_argument('-f', '--feature', type=str,
                        dest='feature', default='gender',
                        help='feature to plot learning curves for')
    parser.add_argument('-r', '--recipe', type=str,
                        dest='recipe',
                        help='path to the recipe to use, if not specified '
                             'default recipe is used')

    args = parser.parse_args()
    infolder = args.infolder
    task = args.feature
    recipe = args.recipe

    print 'Loading dataset...'
    data = ProfilingDataset(infolder)
    print 'Loaded {} users...\n'.format(len(data.entries))
    config = data.config
    tasks = config.tasks
    if task in tasks:
        print ('Creating learning curves for %s task..' % task)
        if not recipe:
            recipe = config.recipes[task]
            clf = from_recipe(recipe)
        else:
            clf = from_recipe(recipe)
        print ('Loading recipe from file %s..' % recipe)
        X, y = data.get_data(task)
        # Cross validation with 100 iterations to get smoother mean test and train
        # score curves, each time with 20% data randomly selected as a validation set.
        cv = cross_validation.KFold(len(X), n_folds=5, random_state=0)
        # cv = cross_validation.ShuffleSplit(len(X), n_iter=20, test_size=0.2,
        #                                    random_state=0)
        title = 'Learning Curves from recipe %s' % recipe
        plot_learning_curve(clf, title, X, y, ylim=(0.3, 1.01), cv=cv, n_jobs=-1)
        plt.show()
    else:
        print('task "%s" does not exist - try one of the'
              ' following: %s' % (task, tasks))
