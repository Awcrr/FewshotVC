import os
import torch
import pickle
import numpy as np
from os import path
from scipy import ndimage as ndi
from torch.nn import functional as F

def create_classifier(args):
    print "=> Creating the {0}_classifier".format(args.classifier)
    classifier = globals()[args.classifier + 'Classifier'](args)
    return classifier

class Classifier(object):
    def __init__(self, args):
        self.n_classes = args.num_classes
        self.n_vcs = args.n_vcs
        self.shots = args.shots
        self.eps = 1e-10
        self.dx = [-1, -1, -1, 0, 1, 1, 1, 0]
        self.dy = [-1, 0, 1, 1, 1, 0, -1, -1]
        self.classifier = args.classifier

    def classify(self, val_dir, binary_train, targets_train, binary_val, targets_val):
        pass

    def calc_stats(self, val_dir, answer, targets_val):
        # Total hit
        result = answer == targets_val
        hit = result.sum()

        stats = 1. * hit / targets_val.shape[0] * 100.
        return stats

class SmoothClassifier(Classifier):
    def __init__(self, args):
        super(SmoothClassifier, self).__init__(args)
        self.sigma = args.sigma

    def classify(self, val_dir, binary_train, targets_train, binary_val, targets_val):
        binary_train = binary_train.reshape(-1, 10, 10, self.n_vcs).transpose((0, 3, 1, 2))
        templates = self._gen_templates(val_dir, binary_train, targets_train)
        templates = templates.reshape(1, templates.shape[0], -1)
        binary_val = binary_val.reshape(-1, 10, 10, self.n_vcs).transpose((0, 3, 1,
            2)).reshape(-1, 1, self.n_vcs * 10 * 10)

        # Calc likelihood
        loglike = binary_val * np.log(templates) + (1 - binary_val) * np.log(1 -
                templates)
        loglike = np.sum(loglike, axis=-1)
        answer = np.argmax(loglike, axis=-1)

        return self.calc_stats(val_dir, answer, targets_val)

    def _gen_templates(self, val_dir, few_set, targets):
        if val_dir != None and path.exists(path.join(val_dir, "templates.npy")):
            print "=> Loading templates"
            return np.load(path.join(val_dir, "templates.npy"))

        print "=> Generating templates"
        temp = np.zeros((self.n_classes, self.n_vcs, 10, 10), dtype=float)
        new_temp = np.zeros((self.n_classes, self.n_vcs, 10, 10), dtype=float)
        for i in xrange(self.n_classes):
            cls = few_set[i * self.shots : (i + 1) * self.shots]
            cls = cls.sum(0)
            cls = cls.astype(float) / self.shots
            temp[targets[i * self.shots]] = cls
         
        for i in xrange(self.n_classes):
            for v in xrange(self.n_vcs):
                new_temp[i, v, :, :] = ndi.filters.gaussian_filter(temp[i][v], self.sigma)

        new_temp = new_temp + self.eps
        if new_temp.max() >= 1.0:
            new_temp *= (1 - self.eps) / new_temp.max()

        if val_dir != None:
            np.save(path.join(val_dir, "templates.npy"), new_temp)

        return new_temp

class NNClassifier(Classifier):
    def __init__(self, args):
        super(NNClassifier, self).__init__(args)

    def classify(self, val_dir, binary_train, targets_train, binary_val, targets_val):
        binary_train = binary_train.reshape(self.n_classes, self.shots, 10, 10, self.n_vcs).transpose((0, 1, 4, 2, 3))
        binary_val = binary_val.reshape(-1, 10, 10, self.n_vcs).transpose((0, 3, 1, 2))
        print "=> Nearest Neighbor for Testing"
        answer = np.zeros(targets_val.shape[0], dtype=targets_val.dtype)
        for i in xrange(targets_val.shape[0]):
            print "| case [{0}]/[{1}]".format(i + 1, targets_val.shape[0])
            score = np.zeros((self.n_classes, self.shots), dtype=float)
            for j in xrange(score.shape[0]):
                for k in xrange(score.shape[1]):
                    score[j][k] = (self._sim(binary_train[j][k], binary_val[i]) + 
                            self._sim(binary_val[i], binary_train[j][k])) * 0.5
            score = score.max(axis=1)
            answer[i] = score.argmax(axis=0)

        return self.calc_stats(val_dir, answer, targets_val)

    def _sim(self, a, b):
        a = F.max_pool2d(torch.Tensor(a).cuda(), kernel_size=5, stride=1, padding=2).data.cpu().numpy().astype(b.dtype)
        similar = np.logical_and(a, b)
        return 1. * similar.sum() / (b.sum() + self.eps)
