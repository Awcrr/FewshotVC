import os
import torch
import pickle
import numpy as np
from os import path
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
from sklearn.cluster import KMeans
from vMFM import VMFM

STRIDE = 8 # stride size
PADDING = 14 # padding size
RF = 36 # receptive field size

def create_extractor(args):
    print "=> Creating the {0}_extractor".format(args.extract)
    extractor = globals()[args.extract + 'Extractor'](args)
    return extractor

class Extractor(object):
    def __init__(self, args):
        self.inter = args.inter
        if self.inter != None and not path.isdir(self.inter):
            os.mkdir(self.inter)
        self.save_inter = args.save_inter
        if self.save_inter != None and not path.isdir(self.save_inter):
            os.mkdir(self.save_inter)
        self.net = torch.load(args.net).cuda().eval()
        self.offset = args.offset if args.offset != None else int(np.ceil(1. * PADDING / STRIDE))
        self.n_vcs = args.n_vcs
        self.sample = args.sample
        self.sample_per_image = args.sample_per_image
        self.center = True

    def set_mode(self, cluster=False):
        if cluster:
            self.offset = int(np.ceil(1. * PADDING / STRIDE))
            self.center = False
        else:
            self.offset = 0
            self.center = True

    def get_features(self, inter, save, img_list):
        if inter != None:
            if not path.isdir(inter):
                os.mkdir(inter)
            feature_path = path.join(inter, "features.npy")
            if path.exists(feature_path):
                print "=> Loading existing pool3 features"
                features = np.load(feature_path)
                patches = None
                locations = None
            else:
                print "[!] Not found: saved features"
                inter = None
                features, patches, locations = self._extract_pool3(img_list)
        else:
            features, patches, locations = self._extract_pool3(img_list)

        if inter == None and save != None:
            if not path.isdir(save):
                os.mkdir(save)
            print "=> Saving pool3 features"
            np.save(path.join(save, "features.npy"), features)
            if patches != None:
                print "=> Saving VC patches"
                with open(path.join(save, "patches.pkl"), 'wb') as f:
                    pickle.dump(patches, f)
            if locations != None:
                print "=> Saving VC locations"
                np.save(path.join(save, "locations.npy"), locations)

        return features

    def get_clusters(self, inter, save, features):
        if inter != None:
            center_path = path.join(inter, "centers.npy")
            if path.exists(center_path):
                print "=> Loading existing visual concepts"
                centers = np.load(center_path)
            else:
                print "[!] Not found: saved centers"
                inter = None
                centers = self._cluster(features)
        else:
            centers = self._cluster(features)

        if inter == None and save != None:
            if not path.isdir(save):
                os.mkdir(save)
            print "=> Saving Visual Concepts(cluster centers)"
            np.save(path.join(save, "centers.npy"), centers)

        return centers

    def draw_examples(self, id, centers=None, features=None, k=15):
        print "=> Drawing examples"
        base_dir = self.inter or self.save_inter 
        base_dir = path.join(base_dir, "centers_{0}".format(id))
        if features is None:
            features = np.load(path.join(base_dir, "features.npy"))
        if centers is None:
            centers = centers or np.load(path.join(base_dir, "centers.npy"))
        with open(path.join(base_dir, "patches.pkl",), 'rb') as f:
            img_list = pickle.load(f)
        locations = np.load(path.join(base_dir, "locations.npy"))

        example_path = path.join(base_dir, "examples")
        if not path.isdir(path.join(base_dir, "examples")):
            os.mkdir(path.join(base_dir, "examples"))

        # Normalize features
        feature_norms = np.sqrt(np.sum(features ** 2, 1)).reshape(-1, 1)
        features = features / feature_norms
        
        # Preprocess images
        trans = transforms.Compose([
            transforms.Resize(84)])
            # transforms.CenterCrop(224)])

        # Extract Top k Examples for each VC:
        for i in xrange(centers.shape[0]):
            print "| Drawing exmples of VC_{0}".format(i)
            center = centers[i]
            # error = np.sum((features - center) ** 2, 1) 
            error = self._get_error(features, center)
            idex = np.argsort(error)[:k]

            big_img = np.zeros((10 + (RF + 10) * 3, 10 + (RF + 10) * 5, 3), dtype=np.uint8)
            for j in xrange(k):
                idx = idex[j]
                img = self._img_loader(img_list[idx])
                img = trans(img)
                img = np.array(img)
                loc = locations[idx]
                h_pos, w_pos = np.unravel_index(j, (3, 5))
                h_pos = h_pos * (10 + RF) + 10
                w_pos = w_pos * (10 + RF) + 10
                big_img[h_pos:h_pos + RF, w_pos:w_pos + RF, :] = img[loc[0]:loc[1], loc[2]:loc[3], :]

            big_img = Image.fromarray(big_img, mode="RGB")
            big_img.save(path.join(example_path, "Examples_VC_{0}.png".format(i)))

    def _img_loader(self, pth):
        with open(pth, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')
 
    def _extract_pool3(self, img_list):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225])
        if self.center:
            trans = transforms.Compose([
                transforms.Scale(84),
                transforms.CenterCrop(84),
                transforms.ToTensor(),
                normalize])
        else:
            trans = transforms.Compose([
                transforms.Scale(84),
                transforms.ToTensor(),
                normalize])

        print "=> Extracting pool3 features"
        total = len(img_list)
        processed = 0
        features = None
        patch_list = []
        location_list = []

        for img in img_list:
            print "| Image [{0}/{1}]".format(processed + 1, total)
            pic = self._img_loader(img)
            pic = trans(pic)
            pic_var = Variable(pic.view(1, pic.size(0), pic.size(1), -1).cuda())

            pool3_var = self.net(pic_var)
            pool3 = pool3_var.data.cpu()
            # Batch to single case
            pool3 = pool3.view(pool3.size(1), pool3.size(2), -1).numpy()
            # Get the original feature map size
            height, width = pool3.shape[1:]
            # Crop the original image part
            pool3 = pool3[:, self.offset:pool3.shape[1] - self.offset, self.offset:pool3.shape[2] - self.offset]
            # Resize to raw feature vectors
            pool3 = pool3.reshape(pool3.shape[0], -1).transpose((1, 0))
            
            if features is None:
                # features = np.zeros((0, pool3.shape[1]), dtype=float)
                features = []

            if self.sample_per_image != None:
                index = np.random.permutation(pool3.shape[0])[: self.sample_per_image]
            else:
                index = np.arange(pool3.shape[0])

            # features = np.vstack((features, pool3[index, :]))
            features += pool3[index, :].tolist()

            # Project feature maps into original image patches
            for pos in index:
                # Calculate the pos in cropped feature map 
                h_pos, w_pos = np.unravel_index(pos, (height - 2 * self.offset, width - 2 * self.offset))
                # Calculate the pos in original feature map
                h_orig = STRIDE * (h_pos + self.offset) - PADDING
                w_orig = STRIDE * (w_pos + self.offset) - PADDING
                if self.offset != None:
                    h_orig = max(h_orig, 0)
                    h_orig = min(h_orig, pic.size(1) - RF)
                    w_orig = max(w_orig, 0)
                    w_orig = min(w_orig, pic.size(2) - RF)
                else:
                    assert h_orig >= 0, "h_orig < 0"
                    assert h_orig <= pic.size(1) - RF, "h_orig is out of bottom bound"
                    assert w_orig >= 0, "w_orig < 0"
                    assert w_orig <= pic.size(2) - RF, "w_orig is out of right bound"
                if self.sample_per_image != None:
                    location_list.append([h_orig, h_orig + RF, w_orig, w_orig + RF])
                else:
                    location_list.append([h_orig, h_orig + RF, w_orig, w_orig + RF, height - 2 * self.offset, width - 2 * self.offset])

                patch_list.append(img)

            processed += 1

        # return features, patch_list, location_list
        return np.array(features, dtype=float), patch_list, location_list

    def _cluster(self, features):
        pass

    def _get_error(self, features, center):
        pass

class VMFMExtractor(Extractor):
    def __init__(self, args):
        super(VMFMExtractor, self).__init__(args)

    def _cluster(self, features):
        print "=> Clustering the features"    
        feature_norms = np.sqrt(np.sum(features ** 2, 1)).reshape(-1, 1)
        features = features / feature_norms

        vmfm = VMFM(cls_num=self.n_vcs, init_method='k++')
        if self.sample is None:
            vmfm.fit(features, kappa=628, max_it=9000, tol=1e-8, normalized=True,
                    verbose=True)
        else:
            perm = np.random.permutation(features.shape[0])
            vmfm.fit(features[perm[:self.sample]], kappa=628, max_it=300, tol=1e-8,
                    normalized=True, verbose=True)
        centers = vmfm.mu

        return centers

    def _get_error(self, features, center):
        center_norm = np.sqrt(np.sum(center ** 2, 0))
        return np.dot(features, center) / (-center_norm) + 1.
