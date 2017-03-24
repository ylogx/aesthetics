"""
Fisher Vector implementation using cv2 v3.2.0+ and python3.

References used below:
[1]: Image Classification with the Fisher Vector: https://hal.inria.fr/file/index/docid/830491/filename/journal.pdf
[2]: http://www.vlfeat.org/api/gmm-fundamentals.html
"""

import glob

import click
import cv2  # v3.2.0
import numpy as np
from scipy.stats import multivariate_normal
from sklearn import svm


class Gmm:
    """ K-component Gaussian Mixture Model """
    def __init__(self, K):
        """ As described in section 2.2, para 3 of reference [1] """
        self.K = K
        self.means = None
        """ Mean Vector """
        self.covariances = None
        """ Covariance Matrix """
        self.weights = None
        """ Mixture Weights """

    def generate(self, input_folder):
        descriptor = Descriptors()
        words = np.concatenate([descriptor.folder(folder) for folder in glob.glob(input_folder + '/*')])
        print("Training GMM of size", self.K)
        self.means, self.covariances, self.weights = self.dictionary(words, self.K)
        # Throw away gaussians with weights that are too small:
        self.means = self.remove_too_small(self.means, self.weights)
        self.covariances = self.remove_too_small(self.covariances, self.weights)
        self.weights = self.remove_too_small(self.weights, self.weights)

        self.save()
        return self.means, self.covariances, self.weights

    def remove_too_small(self, values, weights):
        threshold = 1.0 / self.K
        return np.float32([m for k, m in zip(range(0, len(weights)), values) if weights[k] > threshold])

    def load(self, folder=''):
        import os
        files = ['means.gmm.npy', 'covariances.gmm.npy', 'weights.gmm.npy']
        self.means, self.covariances, self.weights = map(lambda file: np.load(file), map(lambda s: os.path.join(folder, s), files))

    def save(self):
        np.save("means.gmm", self.means)
        np.save("covariances.gmm", self.covariances)
        np.save("weights.gmm", self.weights)

    @staticmethod
    def dictionary(descriptors, K):
        """ See reference [2] """
        em = cv2.ml.EM_create()
        em.setClustersNumber(K)
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
        # _ , descriptors = cv2.xfeatures2d.SIFT_create().detectAndCompute(img, None)
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
        """
        :param samples: X
        :param gmm: Gmm
        :return: np.array fisher vector
        """
        means, covariances, w = gmm.means, gmm.covariances, gmm.weights
        s0, s1, s2 = FisherVector._likelihood_statistics(samples, means, covariances, w)
        T = samples.shape[0]
        diagonal_covariances = np.float32([np.diagonal(covariances[k]) for k in range(0, covariances.shape[0])])
        """ Refer page 4, first column of reference [1] """
        weights = FisherVector._fisher_vector_weights(s0, s1, s2, means, diagonal_covariances, w, T)
        means = FisherVector._fisher_vector_means(s0, s1, s2, means, diagonal_covariances, w, T)
        sigma = FisherVector._fisher_vector_sigma(s0, s1, s2, means, diagonal_covariances, w, T)
        # FIXME: Weights are one dimensional here.
        # fv = np.concatenate([np.concatenate(weights), np.concatenate(means), np.concatenate(sigma)])
        fv = np.concatenate([weights, np.concatenate(means), np.concatenate(sigma)])

        fv = FisherVector.normalize(fv)     # TODO: Normalizing before removing zeros

        return fv


    @staticmethod
    def _likelihood_statistics(samples, means, covariances, weights):
        """
        :param samples: X
        :return: 0th order, 1st order, 2nd order statistics 
                 as described by equation 20, 21, 22 in reference [1]
        """
        def likelihood_moment(x, posterior_probability, moment):
            x_moment = np.power(np.float32(x), moment) if moment > 0 else np.float32([1])
            return x_moment * posterior_probability

        statistics_0_order, statistics_1_order, statistics_2_order = {}, {}, {}
        samples = zip(range(0, len(samples)), samples)

        g = [multivariate_normal(mean=means[k], cov=covariances[k]) for k in range(0, len(weights))]

        gaussians = {index: np.array([g_k.pdf(x) for g_k in g]) for index, x in samples}
        """ u(x) for equation 15, page 4 in reference 1 """

        for k in range(0, len(weights)):
            statistics_0_order[k], statistics_1_order[k], statistics_2_order[k] = 0, 0, 0
            for index, x in samples:
                posterior_probability = FisherVector.posterior_probability(gaussians[index], weights)
                statistics_0_order[k] = statistics_0_order[k] + likelihood_moment(x, posterior_probability[k], 0)
                statistics_1_order[k] = statistics_1_order[k] + likelihood_moment(x, posterior_probability[k], 1)
                statistics_2_order[k] = statistics_2_order[k] + likelihood_moment(x, posterior_probability[k], 2)

        return statistics_0_order, statistics_1_order, statistics_2_order

    @staticmethod
    def posterior_probability(u_gaussian, weights):
        """ Implementation of equation 15, page 4 from reference [1] """
        probabilities = np.multiply(u_gaussian, weights)
        probabilities = probabilities / np.sum(probabilities)
        return probabilities

    @staticmethod
    def _fisher_vector_weights(statistics_0_order, s1, s2, means, covariances, w, T):
        """ Implementation of equation 31, page 6 from reference [1] """
        return np.float32([((statistics_0_order[k] - T * w[k]) / np.sqrt(w[k])) for k in range(0, len(w))])


    @staticmethod
    def _fisher_vector_means(s0, statistics_1_order, s2, means, sigma, w, T):
        """ Implementation of equation 32, page 6 from reference [1] """
        return np.float32([(statistics_1_order[k] - means[k] * s0[k]) /
                           (np.sqrt(w[k] * sigma[k])) for k in range(0, len(w))])


    @staticmethod
    def _fisher_vector_sigma(s0, s1, statistics_2_order, means, sigma, w, T):
        """ Implementation of equation 33, page 6 from reference [1] """
        return np.float32([(statistics_2_order[k] - 2 * means[k] * s1[k] + (means[k] * means[k] - sigma[k]) * s0[k]) /
                           (np.sqrt(2 * w[k]) * sigma[k]) for k in range(0, len(w))])


    @staticmethod
    def normalize(fisher_vector):
        """ Power normalization based on equation 30, page 5, last para; and
        is used in step 3, algorithm 1, page 6 of reference [1] """
        v = np.sign(fisher_vector) * np.sqrt(abs(fisher_vector))    # Power normalization
        return v / np.sqrt(np.dot(v, v))    # L2 Normalization


def train(features):
    # import ipdb as pdb; pdb.set_trace()
    feature_values = list(features.values())
    X = np.concatenate(feature_values)
    Y = np.concatenate([np.float32([i] * len(v)) for i, v in zip(range(0, len(features)), feature_values)])

    clf = svm.SVC()
    clf.fit(X, Y)
    return clf


def success_rate(classifier, features):
    print("Applying the classifier...")
    feature_values = list(features.values())
    X = np.concatenate(feature_values)
    Y = np.concatenate([np.float32([i] * len(v)) for i, v in zip(range(0, len(feature_values)), feature_values)])
    res = float(sum([a == b for a, b in zip(classifier.predict(X), Y)])) / len(Y)
    return res


@click.command()
@click.option('-d', '--dir', default='.', help='Directory of images (default: ./)')
@click.option('-g', '--loadgmm', default=False, is_flag=True, help='Load gmm dictionary from pickles')
@click.option('-n', '--number', default=5, help='Number of words in gmm dictionary')
def main(dir, loadgmm, number):
    print(dir, loadgmm, number)
    gmm = Gmm(K=number)
    if loadgmm:
        gmm.load()
    else:
        gmm.generate(input_folder=dir)

    fisher_vector = FisherVector(gmm)
    features = fisher_vector.features(dir)
    # TBD, split the features into training and validation
    classifier = train(features)
    rate = success_rate(classifier, features)
    print("Success rate is", rate)
    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main())
