import os
import cv2
import sys
import numpy as np
from torch.utils import data
sys.path.append('..')


class PFLDDatasets(data.Dataset):
    def __init__(self, file_list, transforms=None, img_root=None, img_size=112):
        assert img_root is not None
        self.line = None
        self.path = None
        self.img_size = img_size
        self.landmarks = None
        self.filenames = None
        self.euler_angle = None
        self.img_root = img_root
        self.transforms = transforms
        with open(file_list, 'r') as f:
            self.lines = f.readlines()
        
    def __getitem__(self, index):
        self.line = self.lines[index].strip().split()
        self.img = cv2.imread(os.path.join(self.img_root, self.line[0]))
        self.img = cv2.resize(self.img, (self.img_size, self.img_size))
        self.landmark = np.asarray(self.line[1:213], dtype=np.float32)
        self.euler_angle = np.asarray(self.line[213:], dtype=np.float32)
        if self.transforms:
            self.img = self.transforms(self.img)
        return self.img, self.landmark, self.euler_angle

    def __len__(self):
        return len(self.lines)