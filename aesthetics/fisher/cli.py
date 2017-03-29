# -*- coding: utf-8 -*-
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
@click.option('-g', '--loadgmm', default=False, is_flag=True, help='Load gmm dictionary from pickles')
@click.option('-n', '--number', default=5, help='Number of words in gmm dictionary')
@click.option('-l', '--limit', default=50, help='Number of images to read')
def main(dir, loadgmm, number, limit):
    """
    * Create a GMM using the training images.
    * Use this GMM to create feature vectors of training images.
    * Train an SVM on training images.
    * Predict using SVM on training images.
    """
    from aesthetics.fisher import Gmm
    from aesthetics.fisher import FisherVector
    print(dir, loadgmm, number)
    gmm = Gmm(K=number)
    if loadgmm:
        gmm.load()
    else:
        gmm.generate(input_folder=dir, limit=limit)

    fisher_vector = FisherVector(gmm)
    features = fisher_vector.features(dir, limit)
    # TBD, split the features into training and validation
    classifier = train(features)
    rate = success_rate(classifier, features)
    print("Success rate is", rate)
    return 0


if __name__ == '__main__':
    import logging
    import sys

    logging.basicConfig(level=logging.DEBUG)
    sys.exit(main())

