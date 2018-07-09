# -*- coding: utf-8 -*-
import glob

import numpy as np
import os
import tqdm
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import multivariate_normal


class FisherVector(object):
    """
    Fisher Vector implementation using cv2 v3.2.0+ and python3.

    Key Methods:

    * fisher_vector_of_file(self, filename): Returns the fisher vector for given image file
    * get_fisher_vectors_from_folder(self, folder, limit): Returns fisher vectors for all images in given folder
    * features(self, folder, limit): Returns fisher vectors for all images in subfolders of given folder

    References used below:
    [1]: Image Classification with the Fisher Vector: https://hal.inria.fr/file/index/docid/830491/filename/journal.pdf
    [2]: http://www.vlfeat.org/api/gmm-fundamentals.html
    """
    def __init__(self, gmm):
        """
        :param aesthetics.fisher.Gmm gmm: Trained gmm to be used
        """
        self.gmm = gmm

    def features(self, folder, limit):
        """
        :param str folder: Folder Name
        :param int limit: Number of images to read from each subfolder
        :return: fisher vectors for images in each subfolder of given folder
        :rtype: np.array
        """
        folders = sorted(glob.glob(folder + "/*"))
        features = OrderedDict([(f, self.get_fisher_vectors_from_folder(f, limit)) for f in folders])
        return features

    def get_fisher_vectors_from_folder(self, folder, limit):
        """
        :param str folder: Folder Name
        :param int limit: Number of images to read from each folder
        :return: fisher vectors for images in given folder
        :rtype: np.array
        """
        files = glob.glob(folder + "/*.jpg")[:limit]

        with ProcessPoolExecutor() as pool:
            futures = pool.map(self._worker, files)
            desc = 'Creating Fisher Vectors {} images of folder {}'.format(len(files), os.path.split(folder)[-1])
            futures = tqdm.tqdm(futures, total=len(files), desc=desc, unit='image', ncols=120)
            vectors = [f for f in futures if f is not None and len(f) > 0]
            max_shape = np.array([v.shape[0] for v in vectors]).max()
            vectors = [v for v in vectors if v.shape[0] == max_shape]
        # return np.array(vectors)    # Can't do np.float32, because all images may not have same number of features
        return np.float32(vectors)

    def _worker(self, *arg, **kwargs):
        try:
            return self.fisher_vector_of_file(*arg, **kwargs)
        except Exception as e:
            # import pdb; pdb.post_mortem()
            return None

    def fisher_vector_of_file(self, filename):
        """
        :param str filename: Name of the file
        :return: fisher vector of given file
        :rtype: np.array
        """
        import cv2
        img = cv2.imread(filename)
        return self.fisher_vector_of_img_array(img)

    def fisher_vector_of_img_array(self, img):
        """
        :param np.array img: Img Array generated by cv2.imread
        :return: fisher vector of given file
        :rtype: np.array
        """

        def section_fisher(img_section, full_fisher):
            sec_fisher = self.fisher_vector_of_image_section(img_section)
            if sec_fisher.shape == (0,):
                sec_fisher = np.zeros(full_fisher.shape)
            return sec_fisher

        full_fisher = self.fisher_vector_of_image_section(img)
        x, _, _ = img.shape
        loc_mid = int(x / 3)
        loc_bottom = int(2 * x / 3)
        top_fisher = section_fisher(img[0:loc_mid], full_fisher)
        middle_fisher = section_fisher(img[loc_mid + 1:loc_bottom], full_fisher)
        bottom_fisher = section_fisher(img[loc_bottom + 1:x], full_fisher)
        return np.concatenate((full_fisher, top_fisher, middle_fisher, bottom_fisher))

    def fisher_vector_of_image_section(self, img):
        """
        :param np.array img: Img Array generated by cv2.imread
        :return: fisher vector of given img array
        :rtype: np.array
        """
        from aesthetics.fisher import Descriptors
        descriptors = Descriptors()
        img_descriptors = descriptors.image(img)
        if img_descriptors is not None:
            return self._fisher_vector(img_descriptors)
        else:
            return np.empty(0)

    def _fisher_vector(self, img_descriptors):
        """
        :param img_descriptors: X
        :return: fisher vector
        :rtype: np.array
        """
        means, covariances, weights = self.gmm.means, self.gmm.covariances, self.gmm.weights
        s0, s1, s2 = self._likelihood_statistics(img_descriptors)
        T = img_descriptors.shape[0]
        diagonal_covariances = np.float32([np.diagonal(covariances[k]) for k in range(0, covariances.shape[0])])
        """ Refer page 4, first column of reference [1] """
        g_weights = self._fisher_vector_weights(s0, s1, s2, means, diagonal_covariances, weights, T)
        g_means = self._fisher_vector_means(s0, s1, s2, means, diagonal_covariances, weights, T)
        g_sigma = self._fisher_vector_sigma(s0, s1, s2, means, diagonal_covariances, weights, T)
        fv = np.concatenate([np.concatenate(g_weights), np.concatenate(g_means), np.concatenate(g_sigma)])
        fv = self.normalize(fv)
        return fv

    def _likelihood_statistics(self, img_descriptors):
        """
        :param img_descriptors: X
        :return: 0th order, 1st order, 2nd order statistics
                 as described by equation 20, 21, 22 in reference [1]
        """

        def likelihood_moment(x, posterior_probability, moment):
            x_moment = np.power(np.float32(x), moment) if moment > 0 else np.float32([1])
            return x_moment * posterior_probability

        def zeros(like):
            return np.zeros(like.shape).tolist()

        def likelihood_moment_util(sample, posterior_probability):
            return [likelihood_moment(sample, posterior_probability, 0), likelihood_moment(sample, posterior_probability, 1), likelihood_moment(sample, posterior_probability, 2)]

        means, covariances, weights = self.gmm.means, self.gmm.covariances, self.gmm.weights
        normals = [multivariate_normal(mean=means[k], cov=covariances[k]) for k in range(0, len(weights))]
        """ Gaussian Normals """
        gaussian_pdfs = [np.array([g_k.pdf(sample) for g_k in normals]) for sample in img_descriptors]
        """ u(x) for equation 15, page 4 in reference 1 """

        l = len(weights)
        temp = [None]*l
        for k in range(l):
            posterior_probability = [FisherVector.posterior_probability(gaussian_pdfs[i], weights) for i in range(len(img_descriptors))]
            temp[k] = [likelihood_moment_util(sample, posterior_probability[index][k]) for index,sample in enumerate(img_descriptors)]
            temp[k] = np.array(temp[k])
            temp[k] = temp[k].sum(0)
        return np.array([temp[0][0],temp[1][0]]), np.array([temp[0][1],temp[1][1]]), np.array([temp[0][2],temp[1][2]])

@   staticmethod
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

