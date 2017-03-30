
import glob
from concurrent.futures import ProcessPoolExecutor

import cv2  # v3.2.0
import numpy as np
import tqdm


class Descriptors(object):
    """ Convert image to features"""

    def folder(self, folder, limit):
        files = glob.glob(folder + "/*.jpg")[:limit]
        with ProcessPoolExecutor() as executor:
            futures = executor.map(self.image_file, files)
            futures = tqdm.tqdm(futures, total=len(files), desc='Calculating descriptors')
            descriptors = [f for f in futures]
            # descriptors = [self.image_file(file) for file in files]
        descriptors = list(filter(lambda x: x is not None, descriptors))
        return np.concatenate(descriptors)

    def image_file(self, filename):
        """ Refer section 2.2 of reference [1] """
        img = cv2.imread(filename, 0)
        return self.image(img)

    def image(self, img):
        # img = cv2.resize(img, (256, 256))
        # _ , descriptors = cv2.xfeatures2d.SIFT_create().detectAndCompute(img, None)
        _, descriptors = cv2.ORB_create().detectAndCompute(img, None)
        return descriptors


