import os
import numpy as np
from os import path

IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG',
                '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
                ]
class Sampler(object):
    def __init__(self, args):
        self.n_test = args.n_test
        self.imgs = self._get_imgs(args.img_path, args.list_path)
        self.lib_size = len(self.imgs) 
        self.num_classes = args.num_classes
        self.shots = args.shots

    def get_trail(self):
        cls_list = np.random.permutation(self.lib_size)[:self.num_classes]
        train_list = []
        val_list = []
        for i, cls in enumerate(cls_list):
            img_list = np.random.permutation(len(self.imgs[cls]))[:self.shots + 15]
            for j in xrange(self.shots):
                train_list.append(self.imgs[cls][img_list[j]]) # Append img to train_list
            for j in xrange(self.shots, self.shots + 15):
                val_list.append((self.imgs[cls][img_list[j]], i)) # Append img,label pair to val_list

        return train_list, val_list

    def _get_imgs(self, img_path, list_path):
        img_list = []
        with open(path.join(list_path, "test.csv"), 'r') as f:
            f.readline() # Omit the headings
            for i in xrange(self.n_test):
                cur_list = []
                orig_list = None
                cls = None
                for j in xrange(600):
                    line = f.readline().strip()
                    if cls is None:
                        # Get class and original list
                        cls = line.split(',')[1]
                        orig_list = [img for img in os.listdir(path.join(img_path, cls))
                            if self._is_image(img)]
                        orig_list.sort()
                    id = int(line.split(',')[0].split('.')[0][-5:]) - 1
                    temp_name = line.split(',')[0].split('.')[0] + '.JPEG'
                    if temp_name in orig_list:
                        cur_list.append(path.join(img_path, cls, temp_name))
                    elif id < len(orig_list):
                        cur_list.append(path.join(img_path, cls, orig_list[id]))
                img_list.append(cur_list)

        return img_list

    def _is_image(self, pth):
        return any(pth.endswith(extension) for extension in IMG_EXTENSIONS)
