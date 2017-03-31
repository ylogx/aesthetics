# -*- coding: utf-8 -*-
import logging

import click
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import ensemble


def ordered_dict_to_x_y(features):
    logging.info('Key ordering: %s', list(features.keys()))
    feature_values = list(features.values())
    X = np.concatenate(feature_values)
    Y = np.concatenate([np.float32([i] * len(v)) for i, v in zip(range(0, len(features)), feature_values)])
    return X, Y


def train(features):
    X, Y = ordered_dict_to_x_y(features)
    pd.DataFrame(X).to_csv('features.csv')

    clf = get_classification()
    clf.fit(X, Y)
    pd.to_pickle(clf, 'classification.pkl')
    return clf


def get_classification():
    clf = svm.SVC()
    clf = ensemble.GradientBoostingClassifier()
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
    report = classification_report(y_true=Y, y_pred=y_pred, target_names=list(features.keys()))
    logging.info('Classification Report:\n%s', report)
    return precision_score(y_true=Y, y_pred=y_pred)


def predict_from_url(url, *args, **kwargs):
    import os
    os.system('wget {}'.format(url))
    image_path=os.path.split(url)[-1]
    import cv2
    img = cv2.imread(image_path)
    cv2.imwrite(image_path, cv2.resize(img, (500, 500)))
    return predict_image(image_path=image_path, *args, **kwargs)


def predict_image(classifier, gmm, image_path):
    from aesthetics.fisher import FisherVector
    fv = FisherVector(gmm)
    vector = fv.fisher_vector_of_file(image_path)
    return classifier.predict(vector)


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

