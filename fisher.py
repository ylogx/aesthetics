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


class Gmm:
    """ Gaussian Mixture Model """
    def __init__(self):
        """ As described in section 2.2, para 3 of reference [1] """
        self.means = None
        """ Mean Vector """
        self.covariances = None
        """ Covariance Matrix """
        self.weights = None
        """ Mixture Weights """

    def generate(self, input_folder, N):
        descriptor = Descriptors()
        words = np.concatenate([descriptor.folder(folder) for folder in glob.glob(input_folder + '/*')])
        print("Training GMM of size", N)
        self.means, self.covariances, self.weights = self.dictionary(words, N)
        # Throw away gaussians with weights that are too small:
        self.means = self.remove_too_small(self.means, N, self.weights)
        self.covariances = self.remove_too_small(self.covariances, N, self.weights)
        self.weights = self.remove_too_small(self.weights, N, self.weights)

        self.save()
        return self.means, self.covariances, self.weights

    def remove_too_small(self, values, N, weights):
        threshold = 1.0 / N
        return np.float32([m for k, m in zip(range(0, len(weights)), values) if weights[k] > threshold])

    def load(self, folder=''):
        files = ['means.gmm.npy', 'covariances.gmm.npy', 'weights.gmm.npy']
        self.means, self.covariances, self.weights = map(lambda file: np.load(file), map(lambda s: folder + '/', files))

    def save(self):
        np.save("means.gmm", self.means)
        np.save("covariances.gmm", self.covariances)
        np.save("weights.gmm", self.weights)

    @staticmethod
    def dictionary(descriptors, N):
        """ See reference [2] """
        em = cv2.ml.EM_create()
        em.setClustersNumber(N)
        em.trainEM(descriptors)
        return np.float32(em.getMeans()), np.float32(em.getCovs()), np.float32(em.getWeights())[0]


class Descriptors:
    """ Convert image to features"""
    def folder(self, folder):
        files = glob.glob(folder + "/*.jpg")
        print("Calculating descriptors. Number of images is", len(files))
        return np.concatenate([self.image(file) for file in files])

    def image(self, filename):
        """ Refer section 2.2 of reference [1] """
        img = cv2.imread(filename, 0)
        # img = cv2.resize(img, (256, 256))
        # _ , descriptors = cv2.SIFT().detectAndCompute(img, None)
        _, descriptors = cv2.ORB_create().detectAndCompute(img, None)
        print(filename, descriptors)
        return descriptors


class FisherVector:
    def __init__(self, gmm):
        self.gmm = gmm

    def features(self, folder):
        folders = glob.glob(folder + "/*")
        features = {f: self.get_fisher_vectors_from_folder(f, self.gmm) for f in folders}
        return features


    @staticmethod
    def get_fisher_vectors_from_folder(folder, gmm):
        files = glob.glob(folder + "/*.jpg")
        descriptors = Descriptors()
        return np.float32([FisherVector._fisher_vector(descriptors.image(file), gmm) for file in files])


    @staticmethod
    def _fisher_vector(samples, gmm):
        means, covariances, w = gmm.means, gmm.covariances, gmm.weights
        s0, s1, s2 = FisherVector._likelihood_statistics(samples, means, covariances, w)
        T = samples.shape[0]
        diagonal_covariances = np.float32([np.diagonal(covariances[k]) for k in range(0, covariances.shape[0])])
        """ Refer page 4, first column of reference [1] """
        a = FisherVector._fisher_vector_weights(s0, s1, s2, means, diagonal_covariances, w, T)
        b = FisherVector._fisher_vector_means(s0, s1, s2, means, diagonal_covariances, w, T)
        c = FisherVector._fisher_vector_sigma(s0, s1, s2, means, diagonal_covariances, w, T)
        fv = np.concatenate([np.concatenate(a), np.concatenate(b), np.concatenate(c)])
        fv = FisherVector.normalize(fv)
        return fv


    @staticmethod
    def _likelihood_statistics(samples, means, covariances, weights):
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


    @staticmethod
    def _fisher_vector_weights(s0, s1, s2, means, covariances, w, T):
        return np.float32([((s0[k] - T * w[k]) / np.sqrt(w[k])) for k in range(0, len(w))])


    @staticmethod
    def _fisher_vector_means(s0, s1, s2, means, sigma, w, T):
        return np.float32([(s1[k] - means[k] * s0[k]) / (np.sqrt(w[k] * sigma[k])) for k in range(0, len(w))])


    @staticmethod
    def _fisher_vector_sigma(s0, s1, s2, means, sigma, w, T):
        return np.float32([(s2[k] - 2 * means[k] * s1[k] + (means[k] * means[k] - sigma[k]) * s0[k]) /
                           (np.sqrt(2 * w[k]) * sigma[k]) for k in range(0, len(w))])


    @staticmethod
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

    gmm = Gmm()
    if args.loadgmm:
        gmm.load(folder=working_folder)
    else:
        gmm.generate(input_folder=working_folder, N=args.number)

    fisher_vector = FisherVector(gmm)
    features = fisher_vector.features(working_folder)
    # TBD, split the features into training and validation
    classifier = train(features)
    rate = success_rate(classifier, features)
    print("Success rate is", rate)
    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main())
