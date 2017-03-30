# -*- coding: utf-8 -*-
import logging

import click
import numpy as np
from sklearn import svm

def train(features):
    # import ipdb as pdb; pdb.set_trace()
    feature_values = list(features.values())
    X = np.concatenate(feature_values)
    import pandas as pd
    pd.DataFrame(X).to_csv('features.csv')
    Y = np.concatenate([np.float32([i] * len(v)) for i, v in zip(range(0, len(features)), feature_values)])

    clf = svm.SVC()
    clf.fit(X, Y)
    return clf


def success_rate(classifier, features):
    print("Applying the classifier...")
    feature_values = list(features.values())
    X = np.concatenate(feature_values)
    Y = np.concatenate([np.float32([i] * len(v)) for i, v in zip(range(0, len(feature_values)), feature_values)])
    y_pred = classifier.predict(X)
    print('predictions:', list(zip(Y, y_pred)))
    res = float(sum([a == b for a, b in zip(y_pred, Y)])) / len(Y)
    return res


@click.command()
@click.option('-d', '--dir', default='.', help='Directory of images (default: ./)')
@click.option('-g', '--load-gmm', default=False, is_flag=True, help='Load gmm dictionary from pickles')
@click.option('-n', '--number', default=5, help='Number of words in gmm dictionary')
@click.option('-l', '--limit', default=50, help='Number of images to read')
@click.option('-v', '--validation-dir', default=None, help='Directory of images (default: None)')
def main(dir, load_gmm, number, limit, validation_dir):
    """
    * Create a GMM using the training images.
    * Use this GMM to create feature vectors of training images.
    * Train an SVM on training images.
    * Predict using SVM on training images.
    """
    from aesthetics.fisher import Gmm
    from aesthetics.fisher import FisherVector
    logging.debug('dir=%s, load_gmm=%s, number=%s, limit=%s, val_dir=%s', dir, load_gmm, number, limit, validation_dir)
    gmm = Gmm(K=number)
    if load_gmm:
        gmm.load()
    else:
        gmm.generate(input_folder=dir, limit=limit)

    fisher_vector = FisherVector(gmm)
    features = fisher_vector.features(dir, limit)
    # TBD, split the features into training and validation
    classifier = train(features)
    rate = success_rate(classifier, features)
    logging.info("Self test success rate is", rate)
    if validation_dir is not None:
        validation_features = fisher_vector.features(validation_dir, limit)
        rate = success_rate(classifier, validation_features)
        logging.info("Validation test success rate is", rate)
    return 0


if __name__ == '__main__':
    import sys

    logging.basicConfig(level=logging.DEBUG)
    sys.exit(main())

