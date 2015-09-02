#!/usr/bin/python

import sys
import getopt
import math
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from pan import ProfilingDataset
from sklearn.externals import joblib
# The script args
outfolder = ''
infolder = ''
modelfile = ''

try:
    opts, args = getopt.getopt(sys.argv[1:], "hi:o:m:",
                               ["help", "input=", "output="])
except getopt.GetoptError:
    print 'invalid flag or missing option after it, try -h or --help for help'
    sys.exit()

for opt, arg in opts:
    if opt in ('-h', '--help'):
        print '                                                     '
        print '                                                     '
        print ' Hello from the test script. He are some options.'
        print '                                                     '
        print '-------------------- List of options -------------------------'
        print '                                                              '
        print '     -h  or --help     < get some help >'
        print '     -i  or --input    < specify path to training folder >'
        print '     -o  or --output   < specify output directory >'
        print '     -m  or --model    < specify trained model to use>'
        print '                                                              '
        print '                                                              '
        sys.exit()
    elif opt in ('-i', '--input'):
        infolder = arg.rstrip('/')
    elif opt in ('-o', '--output'):
        outfolder = arg.rstrip('/')
    elif opt in ('-m', '--model'):
        modelfile = arg.rstrip('/')

if not outfolder:
    print('outfolder not specified, please pass -o or --output argument')
    sys.exit()
if not infolder:
    print('infile not specified, please pass -i or --infile argument')
    sys.exit()
if not modelfile:
    print('modelfile not specified, please pass -m or --model argument')
    sys.exit()


def test_data(dataset, model):
    """ evaluate model on test data

    :dataset: The dataset to evaluate model on
    :model: The trained model to use for prediction

    """

    # TODO find a better way to solve this
    model.train(model.model)
    # need to retrain - it doesn't keep it's state
    predict = model.test(dataset)
    print('predict size is %s' % len(predict))
    Y = dataset.get_labels(model.feature)
    try:
        # if it's classification we measure micro and macro scores
        acc = accuracy_score(Y, predict)
        conf = confusion_matrix(Y, predict, labels=list(set(Y)))
        print 'Accuracy : {}'.format(acc)
        print 'Confusion matrix :\n {}'.format(conf)
    except ValueError:
        # if it's not, we measure mean square root error (regression)
        sqe = mean_squared_error(Y, predict)
        print 'mean squared error : {}'.format(math.sqrt(sqe))
    dataset.set_labels(model.feature, predict)


data = ProfilingDataset(infolder, label='test')
print 'Loaded {} users...\n'.format(len(data.entries))
config = data.config
features = config.classifier_list + config.regression_list
# load all models for features
all_models = joblib.load(modelfile)
# TODO need to make sure different preprocessing isn't being done
if not all(feature in features for feature in all_models.keys()):
    print("The models you are using aren't all specified in the config file")
    print('Did you change the config file after training???!')
    print('Exiting.. try training again.')
    exit(1)
print '\n--------------- Thy time of Judgement ---------------'
for feature in features:
    test_data(data, all_models[feature])
# write output to file
data.write_data(outfolder)
