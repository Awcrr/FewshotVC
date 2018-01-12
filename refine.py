import os
import torch
import numpy as np
from os import path
from PIL import Image
from torchvision import transforms
from toolkit.pool3_converter import VGGPool3

def create_refiner(args):
    print "=> Creating the {0}_Refiner".format(args.refine)
    refiner = globals()[args.refine + 'Refiner'](args)
    return refiner

class Refiner(object):
    def __init__(self, args):
        self.least_cov = args.least_cov

    def refine(self, save, features, centers):
        print "=> Refining features"
        if save != None and path.exists(path.join(save, "refined.npy")):
            return np.load(path.join(save, "refined.npy"))
        refined = self._dis(features, centers)

        if save != None:
            if not path.isdir(save):
                os.mkdir(save)
            print "=> Saving refined features"
            np.save(path.join(save, "refined.npy"), refined) 

        return refined

    def encode(self, save, refined, threshold):
        if save != None and path.exists(path.join(save, "binary.npy")):
            print "=> Loading binary code"
            binary = np.load(path.join(save, "binary.npy"))
            return binary
        else:
            print "=> Calculating binary code"
            binary = (refined < threshold).astype(int)

        if save != None:
            print "=> Saving binary code"
            np.save(path.join(save, "binary.npy"), binary)

        return binary

    def get_threshold(self, save, refined):
        cov = []
        for i in xrange(1, 1000, 1):
            if self._coverage(refined, i / 1000.) >= self.least_cov:
                cov = [i / 1000., self._coverage(refined, i / 1000.),
                        self._fire_rate(refined, i / 1000.)]
                break

        if save != None:
            np.save(path.join(save, "thres.npy"), cov)
        print "=> Got Threshold:%.4f, Coverage:%.4f, Fire_rate:%.4f" % (cov[0], cov[1],
                cov[2])
        return cov[0]

    def _dis(self, features, center):
        pass

    def _coverage(self, refined, threshold):
        return 1. * np.sum(np.max((refined < threshold), 1), 0) / refined.shape[0]
    def _fire_rate(self, refined, threshold):
        return 1. * np.sum(np.sum((refined < threshold), 1), 0) / refined.shape[0]

class CosineRefiner(Refiner):
    def __init__(self, args):
        super(CosineRefiner, self).__init__(args)

    def _dis(self, features, centers):
        # Normalize features
        feature_norms = np.sqrt(np.sum(features ** 2, 1)).reshape(-1, 1)
        features = features / feature_norms
        center_norms = np.sqrt(np.sum(centers ** 2, 1)).reshape(-1)

        # Calculate cosine distance
        refined = np.zeros((features.shape[0], 0))
        for i in xrange(centers.shape[0]):
            print "| Refine by VC_{0}".format(i)
            center = centers[i]
            norm = center_norms[i]
            distance = np.dot(features, center) / (-norm) + 1.
            refined = np.column_stack((refined, distance))

        return refined
