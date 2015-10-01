#!/usr/bin/python

import os
from argparse import ArgumentParser
from sklearn.externals import joblib
from tictacs import from_recipe
from pan import ProfilingDataset
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix


if __name__ == '__main__':
    parser = ArgumentParser(description='Train pan model on pan dataset')
    parser.add_argument('-i', '--input', type=str,
                        required=True, dest='infolder',
                        help='path to folder with pan dataset for a language')
    parser.add_argument('-o', '--output', type=str,
                        required=True, dest='outfolder',
                        help='path to folder where model should be written')

    args = parser.parse_args()
    infolder = args.infolder
    outfolder = args.outfolder

    dataset = ProfilingDataset(infolder)
    print('Loaded {} users...\n'.format(len(dataset.entries)))
    # get config
    config = dataset.config
    tasks = config.tasks
    print('\n--------------- Thy time of Running ---------------')
    all_models = {}
    for task in tasks:
        print('Learning to judge %s..' % task)
        # load data
        X, y = dataset.get_data(task)
        tictac = from_recipe(config.recipes[task])
        all_models[task] = tictac.fit(X, y)
    modelfile = os.path.join(outfolder, '%s.bin' % dataset.lang)
    print('Writing model to {}'.format(modelfile))
    joblib.dump(all_models, modelfile, compress=3)
