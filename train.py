#!/usr/bin/python

import os
from argparse import ArgumentParser
from pan import ProfilingDataset, AuthorProfilingModel
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
# from sklearn.metrics import accuracy_score, confusion_matrix


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


if __name__ == '__main__':
    parser = ArgumentParser(description='Train pan model on pan dataset')
    parser.add_argument('-i', '--input', type=str,
                        required=True, metavar='infolder',
                        help='path to folder with pan dataset for a language')
    parser.add_argument('-o', '--output', type=str,
                        required=True, metavar='outfolder',
                        help='path to folder where model should be written')

    args = parser.parse_args()
    infolder = args.infolder
    outfolder = args.outfolder

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
