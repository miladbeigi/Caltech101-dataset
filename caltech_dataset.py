from torchvision.datasets import VisionDataset
from sklearn.model_selection import train_test_split

from PIL import Image

import os
import os.path
import sys


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def class_number(root):
    class_set = set(os.listdir(root))
    class_set.remove('BACKGROUND_Google')
    return [*class_set]


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(
            root, transform=transform, target_transform=target_transform)

        self.split = split
        self.classes = class_number(root)
        self.data = []
        with open(f'{self.split}.txt', 'r') as f:
            for each_line in f:
                class_label, name = each_line.split('/')
                if class_label == 'BACKGROUND_Google':
                    continue
                else:
                    path = f'{root}/{class_label}/{name}'.replace('\n', '')
                    self.data.append(
                        (pil_loader(path), self.classes.index(class_label), class_label))
        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        image, label = self.data[index][0: 2]

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.data)
        return length