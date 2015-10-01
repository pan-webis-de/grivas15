#!/usr/bin/python

from argparse import ArgumentParser
from pan import ProfilingDataset
from tictacs import from_recipe
from sklearn.grid_search import GridSearchCV

log = []


def cross_val(dataset, task, model, num_folds=4):
    """ train and cross validate a model

    :lang: the language
    :task: the task we want to classify for , ex: age

    """

    X, y = dataset.get_data(task)
    # get parameters for grid search if it exists - else pass empty dict
    params = model.grid_params if hasattr(model, 'grid_params') else dict()
    print '\nCreating model for %s - %s' % (dataset.lang, task)
    print 'Using %s fold validation' % (num_folds)
    # get data
    log.append('\nResults for %s - %s with classifier %s' %
               (dataset.lang, task, model.__class__.__name__))
    if task in dataset.config.classifier_list:
        grid_cv = GridSearchCV(model, params, cv=num_folds, verbose=1,
                               n_jobs=-1)
        grid_cv.fit(X, y)
        accuracy = grid_cv.best_score_
        log.append('best params: %s' % grid_cv.best_params_)
        log.append('Accuracy mean : %s' % accuracy)
    else:
        # if it's not, we measure mean square root error (regression)
        grid_cv = GridSearchCV(model, params, scoring='mean_squared_error',
                               cv=num_folds, verbose=1, n_jobs=-1)
        grid_cv.fit(X, y)
        accuracy = grid_cv.best_score_
        log.append('root mean squared error : %s' % accuracy)

if __name__ == '__main__':
    parser = ArgumentParser(description='Train a model with crossvalidation'
                            ' on pan dataset - used for testing purposes ')
    parser.add_argument('-i', '--input', type=str,
                        required=True, dest='infolder',
                        help='path to folder with pan dataset for a language')
    parser.add_argument('-n', '--numfolds', type=int,
                        dest='num_folds', default=4,
                        help='Number of folds to use in cross validation')

    args = parser.parse_args()
    infolder = args.infolder
    num_folds = args.num_folds

    print('Loading dataset...')
    dataset = ProfilingDataset(infolder)
    print('Loaded %s users...\n' % len(dataset.entries))
    config = dataset.config
    tasks = config.tasks
    print('\n--------------- Thy time of Running ---------------')
    for task in tasks:
        tictac = from_recipe(config.recipes[task])
        cross_val(dataset, task, tictac, num_folds)
    # print results at end
    print('\n--------------- Thy time of Judgement ---------------')
    for message in log:
        print(message)
