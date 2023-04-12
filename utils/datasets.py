import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import json
class LoadImagesAndLabels(Dataset):
    def __init__(self, path, img_size):
        train_path = path + 'train.txt'
        with open(train_path, 'r') as file:
            self.img_files = file.readlines()
        self.img_files = [x.replace('\n', '') for x in self.img_files]
        self.indices = len(self.img_files)
        self.img_size = img_size

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):

        # Load image
        img,label = self.load_image_label(index)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)
        return img, label

    def load_image_label(self, index):
        img_path = self.img_files[index]
        label_path = self.img_files[index].replace('images', 'labels').replace('.jpg', '.json')
        img = cv2.imread(img_path)  # BGR
        img = cv2.resize(img, self.img_size)
        label = json.load(open(label_path, 'r'))
        return img, label
 


def create_dataloader(path, imgsz, batch_size):
    dataset = LoadImagesAndLabels(path, imgsz)

    batch_size = min(batch_size, len(dataset))
    sampler = torch.utils.data.RandomSampler(dataset)
    loader = torch.utils.data.DataLoader
    dataloader = loader(dataset,
                        batch_size=batch_size,
                        num_workers=2,
                        sampler=sampler,
                        pin_memory=True
                        )
    return dataloader, dataset