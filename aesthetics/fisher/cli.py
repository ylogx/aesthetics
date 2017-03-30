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
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import precision_score

    logging.info('Applying the classifier...')
    feature_values = list(features.values())
    X = np.concatenate(feature_values)
    Y = np.concatenate([np.float32([i] * len(v)) for i, v in zip(range(0, len(feature_values)), feature_values)])
    y_pred = classifier.predict(X)
    # logging.debug('Predictions:\n%s', list(zip(Y, y_pred)))
    logging.info('Confusion Matrix:\n%s', confusion_matrix(y_true=Y, y_pred=y_pred))
    report = classification_report(y_true=Y, y_pred=y_pred, target_names=['low', 'high'])
    logging.info('Classification Report:\n%s', report)
    return precision_score(y_true=Y, y_pred=y_pred)


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
    logging.info("Self test success rate is %.2f", rate)
    if validation_dir is not None:
        validation_features = fisher_vector.features(validation_dir, limit)
        rate = success_rate(classifier, validation_features)
        logging.info("Validation test success rate is %.2f", rate)
    return 0


if __name__ == '__main__':
    import sys

    logging.basicConfig(level=logging.DEBUG)
    sys.exit(main())

