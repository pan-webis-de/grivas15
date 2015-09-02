#!/usr/bin/python

import sys
import getopt
import math
from pan import ProfilingDataset, AuthorProfilingModel
# from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, accuracy_score, \
                            f1_score, confusion_matrix, mean_squared_error

infolder = ''

try:
    opts, args = getopt.getopt(sys.argv[1:], "hi:o:",
                               ["help", "input=", "output="])
except getopt.GetoptError:
    print 'invalid flag or missing option after it, try -h or --help for help'
    sys.exit()

for opt, arg in opts:
    if opt in ('-h', '--help'):
        print '                                                     '
        print '                                                     '
        print ' Hello from the training script. He are some options.'
        print '                                                     '
        print '-------------------- List of options -------------------------'
        print '                                                              '
        print '     -h  or --help     < get some help >'
        print '     -i  or --input    < specify path to training folder >'
        print '                                                              '
        print '                                                              '
        sys.exit()
    elif opt in ('-i', '--input'):
        infolder = arg.rstrip('/')

if not infolder:
    print('infile not specified, please pass -i or --infile argument')
    sys.exit()

log = []


def cross_validate(dataset, feature, clf):
    """ train and cross validate a model

    :lang: the language
    :feature: the feature we want to classify for , ex: age

    """

    num_folds = 4
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


print 'Loading dataset...'
data = ProfilingDataset(infolder, label='train')
print 'Loaded {} users...\n'.format(len(data.entries))
config = data.config
features = config.classifier_list + config.regression_list
print '\n--------------- Thy time of Running ---------------'
for feature in features:
    cross_validate(data, feature, clf=config[feature].estimator_model)
# print results at end
print '\n--------------- Thy time of Judgement ---------------'
for message in log:
    print message
print
