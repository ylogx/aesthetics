# -*- coding: utf-8 -*-
from unittest import TestCase
from aesthetics.fisher import FisherVector
from aesthetics.fisher import Gmm
from aesthetics.fisher import Descriptors
from eyeqscore.combined.mapr.job import data_folder_in_mrjob
filename = 'simple.jpeg'

class TestFisherVector(TestCase):

    def _setup(self):
        gmm = Gmm(K=5)
        gmm.load(folder=data_folder_in_mrjob)
        return FisherVector(gmm)

    def test_likelihood_statistics(self):
        fv = self._setup()
        from aesthetics.fisher import Descriptors
        descriptors = Descriptors()
        import cv2
        img = cv2.imread(filename)
        img_descriptors = descriptors.image(img)
        x = fv._likelihood_statistics(img_descriptors)
        self.assertEqual(x,fv._likelihood_statistics(img_descriptors) )
