
import glob

import cv2  # v3.2.0
import numpy as np


class Gmm(object):
    """ K-component Gaussian Mixture Model """

    def __init__(self, K):
        """
        As described in section 2.2, para 3 of reference [1]

        :param K: Number of components
        """
        self.K = K
        self.means = None
        """ Mean Vector """
        self.covariances = None
        """ Covariance Matrix """
        self.weights = None
        """ Mixture Weights """

    def generate(self, input_folder, limit):
        """
        Reads all the images files from the sub-folders of the 'input_folder'
        and creates a gmm definition (mean, covariances, weights) from those
        image's features.

        :param input_folder: Folder whose subfolder contains images
        :type input_folder: str
        :param limit: Number of images to be read
        :type limit: int
        :return: tuple(mean, covariance, weights)
        :rtype: tuple
        """
        from aesthetics.fisher import Descriptors

        descriptor = Descriptors()
        img_descriptors = [descriptor.folder(folder, limit) for folder in sorted(glob.glob(input_folder + '/*'))]
        max_shape = np.array([v.shape[0] for v in img_descriptors]).max()
        img_descriptors = list(filter(lambda x: x is not None and x.shape[0] == max_shape, img_descriptors))
        words = np.concatenate(img_descriptors)
        print("Training GMM of size", self.K)
        self.means, self.covariances, self.weights = self.train_expectation_maximisation(words, self.K)
        # Throw away gaussians with weights that are too small:
        self.means = self._remove_too_small(self.means, self.weights)
        self.covariances = self._remove_too_small(self.covariances, self.weights)
        self.weights = self._remove_too_small(self.weights, self.weights)

        self.save()
        return self.means, self.covariances, self.weights

    def _remove_too_small(self, values, weights):
        threshold = 1.0 / self.K
        return np.float32([m for k, m in zip(range(0, len(weights)), values) if weights[k] > threshold])

    def load(self, folder=''):
        """
        Load gmm definition from self.save'd folder data

        :param folder: Folder name
        :type folder: str
        :return: nothing
        """
        import os
        files = ['means.gmm.npy', 'covariances.gmm.npy', 'weights.gmm.npy']
        self.means, self.covariances, self.weights = map(lambda file: np.load(file),
                                                         map(lambda s: os.path.join(folder, s), files))

    def save(self):
        """
        Save gmm definition to current folder

        :return: nothing
        """
        np.save("means.gmm", self.means)
        np.save("covariances.gmm", self.covariances)
        np.save("weights.gmm", self.weights)

    @staticmethod
    def train_expectation_maximisation(descriptors, K):
        """ See reference [2] """
        em = cv2.ml.EM_create()
        em.setClustersNumber(K)
        em.trainEM(descriptors)
        return np.float32(em.getMeans()), np.float32(em.getCovs()), np.float32(em.getWeights())[0]


