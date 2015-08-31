#!/usr/bin/python

import os
import sys
import getopt
from pan import ProfilingDataset, AuthorProfilingModel
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
# from sklearn.metrics import accuracy_score, confusion_matrix

outfolder = ''
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
        print '     -o  or --output   < specify path to model folder >'
        print '                                                              '
        print '                                                              '
        sys.exit()
    elif opt in ('-i', '--input'):
        infolder = arg.rstrip('/')
    elif opt in ('-o', '--output'):
        outfolder = arg.rstrip('/')

if not outfolder:
    print('outfolder not specified, please pass -o or --output argument')
    sys.exit()
if not infolder:
    print('infile not specified, please pass -i or --infile argument')
    sys.exit()


def modelify(dataset, feature, clf):
    """ train a model for lang and feature, where feature is the
    feature we want to classify for example : age

    :lang: the language
    :feature: the feature we want to classify for , ex: age
    :returns: a binary model of the classifier

    """

    # get data
    print 'Creating model for {} - {}'.format(dataset.lang, feature)
    # classifier = KNeighborsClassifier(n_neighbors=1)
    model = AuthorProfilingModel(dataset=dataset, feature=feature, clf=clf)
    model.train(clf)
    # model.pickle(outfolder)
    return model

data = ProfilingDataset(infolder, label='train')
print 'Loaded {} users...\n'.format(len(data.entries))
config = data.config
features = config.classifier_list + config.regression_list
print '\n--------------- Thy time of Running ---------------'
all_models = {}
for feature in features:
    all_models[feature] = modelify(data, feature,
                                   clf=config[feature].estimator_model)
modelfile = os.path.join(outfolder, '{}.bin'.format(data.lang))
print 'Writing model to {}'.format(modelfile)
joblib.dump(all_models, modelfile, compress=1)
