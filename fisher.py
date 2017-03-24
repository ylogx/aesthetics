"""
Fisher Vector implementation using cv2 v3.2.0+ and python3.

Please refer to the paper:
[1]: Image Classification with the Fisher Vector: https://hal.inria.fr/file/index/docid/830491/filename/journal.pdf
[2]: http://www.vlfeat.org/api/gmm-fundamentals.html
"""

import glob

import cv2  # v3.2.0
import numpy as np
from scipy.stats import multivariate_normal
from sklearn import svm


def load_gmm(folder=""):
    files = ["means.gmm.npy", "covariances.gmm.npy", "weights.gmm.npy"]
    return map(lambda file: np.load(file), map(lambda s: folder + "/", files))


def generate_gmm(input_folder, N):
    """
    Implementation of Gaussian Mixture Model using Expectation Maximisation
    """
    words = np.concatenate([folder_descriptors(folder) for folder in glob.glob(input_folder + '/*')])
    print("Training GMM of size", N)
    means, covariances, weights = dictionary(words, N)
    # Throw away gaussians with weights that are too small:
    th = 1.0 / N
    means = np.float32([m for k, m in zip(range(0, len(weights)), means) if weights[k] > th])
    covariances = np.float32([m for k, m in zip(range(0, len(weights)), covariances) if weights[k] > th])
    weights = np.float32([m for k, m in zip(range(0, len(weights)), weights) if weights[k] > th])

    np.save("means.gmm", means)
    np.save("covariances.gmm", covariances)
    np.save("weights.gmm", weights)
    return means, covariances, weights


def folder_descriptors(folder):
    files = glob.glob(folder + "/*.jpg")
    print("Calculating descriptors. Number of images is", len(files))
    return np.concatenate([image_descriptors(file) for file in files])


def image_descriptors(file):
    """ Refer section 2.2 of reference [1] """
    img = cv2.imread(file, 0)
    # img = cv2.resize(img, (256, 256))
    # _ , descriptors = cv2.SIFT().detectAndCompute(img, None)
    _, descriptors = cv2.ORB_create().detectAndCompute(img, None)
    print(file, descriptors)
    return descriptors


def dictionary(descriptors, N):
    """ See reference [2] """
    em = cv2.ml.EM_create()
    em.setClustersNumber(N)
    em.trainEM(descriptors)
    return np.float32(em.getMeans()), \
           np.float32(em.getCovs()), np.float32(em.getWeights())[0]


# Fisher Vector #

def fisher_features(folder, gmm):
    folders = glob.glob(folder + "/*")
    features = {f: get_fisher_vectors_from_folder(f, gmm) for f in folders}
    return features


def get_fisher_vectors_from_folder(folder, gmm):
    files = glob.glob(folder + "/*.jpg")
    return np.float32([fisher_vector(image_descriptors(file), *gmm) for file in files])


def fisher_vector(samples, means, covariances, w):
    s0, s1, s2 = likelihood_statistics(samples, means, covariances, w)
    T = samples.shape[0]
    diagonal_covariances = np.float32([np.diagonal(covariances[k]) for k in range(0, covariances.shape[0])])
    """ Refer page 4, first column of reference [1] """
    a = _fisher_vector_weights(s0, s1, s2, means, diagonal_covariances, w, T)
    b = _fisher_vector_means(s0, s1, s2, means, diagonal_covariances, w, T)
    c = _fisher_vector_sigma(s0, s1, s2, means, diagonal_covariances, w, T)
    fv = np.concatenate([np.concatenate(a), np.concatenate(b), np.concatenate(c)])
    fv = normalize(fv)
    return fv


def likelihood_statistics(samples, means, covariances, weights):
    def likelihood_moment(x, ytk, moment):
        x_moment = np.power(np.float32(x), moment) if moment > 0 else np.float32([1])
        return x_moment * ytk

    gaussians, s0, s1, s2 = {}, {}, {}, {}
    samples = zip(range(0, len(samples)), samples)

    g = [multivariate_normal(mean=means[k], cov=covariances[k]) for k in range(0, len(weights))]
    for index, x in samples:
        gaussians[index] = np.array([g_k.pdf(x) for g_k in g])

    for k in range(0, len(weights)):
        s0[k], s1[k], s2[k] = 0, 0, 0
        for index, x in samples:
            probabilities = np.multiply(gaussians[index], weights)
            probabilities = probabilities / np.sum(probabilities)
            s0[k] = s0[k] + likelihood_moment(x, probabilities[k], 0)
            s1[k] = s1[k] + likelihood_moment(x, probabilities[k], 1)
            s2[k] = s2[k] + likelihood_moment(x, probabilities[k], 2)

    return s0, s1, s2


def _fisher_vector_weights(s0, s1, s2, means, covariances, w, T):
    return np.float32([((s0[k] - T * w[k]) / np.sqrt(w[k])) for k in range(0, len(w))])


def _fisher_vector_means(s0, s1, s2, means, sigma, w, T):
    return np.float32([(s1[k] - means[k] * s0[k]) / (np.sqrt(w[k] * sigma[k])) for k in range(0, len(w))])


def _fisher_vector_sigma(s0, s1, s2, means, sigma, w, T):
    return np.float32([(s2[k] - 2 * means[k] * s1[k] + (means[k] * means[k] - sigma[k]) * s0[k]) /
                       (np.sqrt(2 * w[k]) * sigma[k]) for k in range(0, len(w))])


def normalize(fisher_vector):
    v = np.sqrt(abs(fisher_vector)) * np.sign(fisher_vector)
    return v / np.sqrt(np.dot(v, v))


def train(features):
    X = np.concatenate(features.values())
    Y = np.concatenate([np.float32([i] * len(v)) for i, v in zip(range(0, len(features)), features.values())])

    clf = svm.SVC()
    clf.fit(X, Y)
    return clf


def success_rate(classifier, features):
    print("Applying the classifier...")
    X = np.concatenate(np.array(features.values()))
    Y = np.concatenate([np.float32([i] * len(v)) for i, v in zip(range(0, len(features)), features.values())])
    res = float(sum([a == b for a, b in zip(classifier.predict(X), Y)])) / len(Y)
    return res


def main():
    def get_args():
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('-d', "--dir", help="Directory with images", default='.')
        parser.add_argument("-g", "--loadgmm", help="Load Gmm dictionary", action='store_true', default=False)
        parser.add_argument('-n', "--number", help="Number of words in dictionary", default=5, type=int)
        args = parser.parse_args()
        return args

    args = get_args()
    working_folder = args.dir

    gmm = load_gmm(working_folder) if args.loadgmm else generate_gmm(working_folder, args.number)
    ffeatures = fisher_features(working_folder, gmm)
    # TBD, split the features into training and validation
    classifier = train(ffeatures)
    rate = success_rate(classifier, ffeatures)
    print("Success rate is", rate)
    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main())
