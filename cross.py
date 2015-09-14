#!/usr/bin/python

import math
from argparse import ArgumentParser
from pan import ProfilingDataset
# from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, accuracy_score, \
                            f1_score, confusion_matrix, mean_squared_error
from sklearn import cross_validation
from tictacs import from_recipe

log = []

def cross_val(dataset, task, model, num_folds=4):
    """ train and cross validate a model

    :lang: the language
    :task: the task we want to classify for , ex: age

    """

    X, y = dataset.get_data(task)
    print '\nCreating model for {} - {}'.format(dataset.lang, task)
    print 'Using {} fold validation'.format(num_folds)
    # get data
    log.append('\nResults for {} - {} with classifier {}'
               .format(dataset.lang, task, model.__class__.__name__))
    if task in dataset.config.classifier_list:
        cv = cross_validation.StratifiedKFold(y, n_folds=num_folds,
                                              random_state=13, shuffle=True)
        predict = cross_validation.cross_val_predict(model, X, y, cv=cv)
        # if it's classification we measure micro and macro scores
        # f1_macro = f1_score(y, predict, average='macro', pos_label=None)
        # f1_micro = f1_score(y, predict, average='micro', pos_label=None)
        # prec_macro = precision_score(y, predict, average='macro', pos_label=None)
        # prec_micro = precision_score(y, predict, average='micro', pos_label=None)
        # rec_macro = recall_score(y, predict, average='macro', pos_label=None)
        # rec_micro = recall_score(y, predict, average='micro', pos_label=None)
        accuracy = accuracy_score(y, predict)
        conf = confusion_matrix(y, predict)
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
        cv = cross_validation.KFold(y, n_folds=num_folds,
                                    random_state=13, shuffle=True)
        predict = cross_validation.cross_val_predict(model, X, y, cv=cv)
        # if it's not, we measure mean square root error (regression)
        sqe = mean_squared_error(y, predict)
        log.append('root mean squared error : {}'.format(math.sqrt(sqe)))

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

    print 'Loading dataset...'
    dataset = ProfilingDataset(infolder)
    print 'Loaded {} users...\n'.format(len(dataset.entries))
    config = dataset.config
    tasks = config.tasks
    print '\n--------------- Thy time of Running ---------------'
    for task in tasks:
        tictac = from_recipe(config.recipes[task])
        cross_val(dataset, task, tictac, num_folds)
    # print results at end
    print '\n--------------- Thy time of Judgement ---------------'
    for message in log:
        print message
