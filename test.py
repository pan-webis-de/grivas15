#!/usr/bin/python

from argparse import ArgumentParser
import math
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from pan import ProfilingDataset
from sklearn.externals import joblib


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


if __name__ == '__main__':
    parser = ArgumentParser(description='Test trained model on pan dataset')
    parser.add_argument('-i', '--input', type=str,
                        required=True, dest='infolder',
                        help='path to folder with pan dataset for a language')
    parser.add_argument('-o', '--output', type=str,
                        required=True, dest='outfolder',
                        help='path to folder where model should be written')
    parser.add_argument('-m', '--model', type=str,
                        required=True, dest='model',
                        help='path to learned model to use for predictions')

    args = parser.parse_args()
    model = args.model
    infolder = args.infolder
    outfolder = args.outfolder

    data = ProfilingDataset(infolder, label='test')
    print 'Loaded {} users...\n'.format(len(data.entries))
    config = data.config
    features = config.classifier_list + config.regression_list
    # load all models for features
    all_models = joblib.load(model)
    # TODO need to make sure different preprocessing isn't being done
    if not all(feature in features for feature in all_models.keys()):
        print("The models you are using aren't all specified in config file")
        print('Did you change the config file after training???!')
        print('Exiting.. try training again.')
        exit(1)
    print '\n--------------- Thy time of Judgement ---------------'
    for feature in features:
        test_data(data, all_models[feature])
    # write output to file
    data.write_data(outfolder)
