
import glob
from concurrent.futures import ProcessPoolExecutor

import cv2  # v3.2.0
import numpy as np
import tqdm


class Descriptors(object):
    """
    Convert image to features

    Key Methods:
    * image(self, img): Given a img array, returns its descriptors
    * image_file(self, filename): Given a image filename, returns its descriptors
    """
    def __init__(self):
        self.feature_transform = None

    def folder(self, folder, limit):
        """
        :param folder: Name of the folder containing images
        :type folder: str
        :param limit: Number of images to be read from given folder
        :type limit: int
        :return: List of descriptors of the given images
        :rtype: np.array
        """
        files = glob.glob(folder + "/*.jpg")[:limit]
        with ProcessPoolExecutor() as executor:
            futures = executor.map(self.image_file, files)
            futures = tqdm.tqdm(futures, total=len(files), desc='Calculating descriptors')
            descriptors = [f for f in futures]
            # descriptors = [self.image_file(file) for file in files]
        descriptors = list(filter(lambda x: x is not None, descriptors))
        return np.concatenate(descriptors)

    def image_file(self, filename):
        """
        Refer section 2.2 of reference [1]

        :param filename: Name of the image to be read
        :type filename: str
        :return: Descriptors of the given image
        :rtype: np.array
        """
        img = cv2.imread(filename, 0)
        if type(img) == np.ndarray:
            return self.image(img)
        else:
            return None

    def image(self, img):
        """
        :param img: Image array read using cv2.imread
        :type img: np.array
        :return: Descriptors of the given image
        :rtype: np.array
        """
        # img = cv2.resize(img, (500, 500))
        if self.feature_transform is None:
            self.feature_transform = cv2.xfeatures2d.SIFT_create()
            # self.feature_transform = cv2.ORB_create()
        _, descriptors = self.feature_transform.detectAndCompute(img, None)
        return descriptors


