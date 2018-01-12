import time
import torch
import numpy as np
from os import path
from opts import args
from vgg_pool3 import VGGPool3
from dataset import Sampler 
from extract import create_extractor
from refine import create_refiner
from classifiers import create_classifier

# Update random seed
np.random.seed(int(time.time()))

print "=> Creating a Sampler"
sampler = Sampler(args)
print "=> Sampler is ready"

print "=> Creating extractor"
extractor = create_extractor(args)
print "=> Extractor is ready"

print "=> Creating refiner"
refiner = create_refiner(args)
print "=> Refiner is ready"

print "=> Creating a classifier"
classifier = create_classifier(args) 
print "=> Classifier is ready"

overall = 0.0
rec = []
for i in xrange(args.trails):
    print "=> Doing trail [{0}/{1}]".format(i + 1, args.trails)
    train_list, val_list = sampler.get_trail() 
    # Process training images
    if not args.classifier.startswith("Baseline"):
        extractor.set_mode(cluster=True)
        train_features = extractor.get_features(None, None, train_list)
        vcs = extractor.get_clusters(None, None, train_features)

    extractor.set_mode(cluster=False)
    train_features = extractor.get_features(None, None, train_list)
    if not args.classifier.startswith("Baseline"):
        train_refined = refiner.refine(None, train_features, vcs)
        threshold = refiner.get_threshold(None, train_refined)
        train_binary = refiner.encode(None, train_refined, threshold)
    train_targets = np.array([k for k in xrange(args.num_classes) for j in xrange(args.shots)])

    # Process validation images
    extractor.set_mode(cluster=False)
    val_imgs = [pr[0] for pr in val_list]
    val_targets = np.array([pr[1] for pr in val_list])
    val_features = extractor.get_features(None, None, val_imgs)
    if not args.classifier.startswith("Baseline"):
        val_refined = refiner.refine(None, val_features, vcs)
        val_binary = refiner.encode(None, val_refined, threshold)
        
    # Classification
    if args.classifier != "Baseline" and args.classifier != "DistanceNN":
        result = classifier.classify(None, train_binary, train_targets, val_binary, val_targets)
    elif args.classifier == "DistanceNN":
        result = classifier.classify(None, train_refined, train_targets, val_refined, val_targets)
    else:
        result = classifier.classify(None, train_features, train_targets, val_features, val_targets)
    rec.append(result)
    overall = (overall * i + result) / (i + 1.)
    print "=> Trail_%d\tResults: %.4f\tOverall: %.4f" % (i + 1, result, overall)
    if args.record != None:
        np.save(args.record, np.array(rec))

print "=> Final result: {0}".format(overall)
