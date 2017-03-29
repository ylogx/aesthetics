"""
Fisher Vector implementation using cv2 v3.2.0+ and python3.

References used below:
[1]: Image Classification with the Fisher Vector: https://hal.inria.fr/file/index/docid/830491/filename/journal.pdf
[2]: http://www.vlfeat.org/api/gmm-fundamentals.html
"""
import glob
import multiprocessing
import os

import numpy as np
from scipy.stats import multivariate_normal


class FisherVector(object):
    def __init__(self, gmm):
        self.gmm = gmm

    def features(self, folder, limit):
        folders = glob.glob(folder + "/*")
        features = {f: self.get_fisher_vectors_from_folder(f, limit) for f in folders}
        return features

    def get_fisher_vectors_from_folder(self, folder, limit):
        files = glob.glob(folder + "/*.jpg")[:limit]

        with multiprocessing.Pool() as pool:
            desc = 'Creating Fisher Vectors in parallel for {} images of folder {}'.format(
                len(files),
                os.path.split(folder)[-1]
            )
            print(desc)
            # files = tqdm.tqdm(files, total=len(files), desc=desc, unit='image')
            vectors = pool.map(self.fisher_vector_of_file, files)
        return np.float32(vectors)

    def fisher_vector_of_file(self, filename):
        import cv2
        img = cv2.imread(filename)
        full_fisher = self.fisher_vector_of_image(img)
        x, _, _ = img.shape
        loc_mid = int(x / 3)
        loc_bottom = int(2 * x / 3)
        top_fisher = self.fisher_vector_of_image(img[0:loc_mid])
        middle_fisher = self.fisher_vector_of_image(img[loc_mid + 1:loc_bottom])
        bottom_fisher = self.fisher_vector_of_image(img[loc_bottom + 1:x])
        return np.concatenate((full_fisher, top_fisher, middle_fisher, bottom_fisher))

    def fisher_vector_of_image(self, img):
        from aesthetics.fisher import Descriptors
        descriptors = Descriptors()
        return self._fisher_vector(descriptors.image(img))

    def _fisher_vector(self, samples):
        """
        :param samples: X
        :return: np.array fisher vector
        """
        means, covariances, weights = self.gmm.means, self.gmm.covariances, self.gmm.weights
        s0, s1, s2 = self._likelihood_statistics(samples)
        T = samples.shape[0]
        diagonal_covariances = np.float32([np.diagonal(covariances[k]) for k in range(0, covariances.shape[0])])
        """ Refer page 4, first column of reference [1] """
        g_weights = self._fisher_vector_weights(s0, s1, s2, means, diagonal_covariances, weights, T)
        g_means = self._fisher_vector_means(s0, s1, s2, means, diagonal_covariances, weights, T)
        g_sigma = self._fisher_vector_sigma(s0, s1, s2, means, diagonal_covariances, weights, T)
        # FIXME: Weights are one dimensional here.
        fv = np.concatenate([np.concatenate(g_weights), np.concatenate(g_means), np.concatenate(g_sigma)])
        # fv = np.concatenate([g_weights, np.concatenate(means), np.concatenate(g_sigma)])
        fv = self.normalize(fv)
        return fv

    def _likelihood_statistics(self, samples):
        """
        :param samples: X
        :return: 0th order, 1st order, 2nd order statistics
                 as described by equation 20, 21, 22 in reference [1]
        """

        def likelihood_moment(x, posterior_probability, moment):
            x_moment = np.power(np.float32(x), moment) if moment > 0 else np.float32([1])
            return x_moment * posterior_probability

        def zeros(like):
            return np.zeros(like.shape).tolist()

        means, covariances, weights = self.gmm.means, self.gmm.covariances, self.gmm.weights
        normals = [multivariate_normal(mean=means[k], cov=covariances[k]) for k in range(0, len(weights))]
        """ Gaussian Normals """
        gaussian_pdfs = [np.array([g_k.pdf(sample) for g_k in normals]) for sample in samples]
        """ u(x) for equation 15, page 4 in reference 1 """
        statistics_0_order, statistics_1_order, statistics_2_order = zeros(weights), zeros(weights), zeros(weights)
        for k in range(0, len(weights)):
            for index, sample in enumerate(samples):
                posterior_probability = FisherVector.posterior_probability(gaussian_pdfs[index], weights)
                statistics_0_order[k] = statistics_0_order[k] + likelihood_moment(sample, posterior_probability[k], 0)
                statistics_1_order[k] = statistics_1_order[k] + likelihood_moment(sample, posterior_probability[k], 1)
                statistics_2_order[k] = statistics_2_order[k] + likelihood_moment(sample, posterior_probability[k], 2)

        return np.array(statistics_0_order), np.array(statistics_1_order), np.array(statistics_2_order)

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
        v = np.sign(fisher_vector) * np.sqrt(abs(fisher_vector))  # Power normalization
        return v / np.sqrt(np.dot(v, v))  # L2 Normalization
