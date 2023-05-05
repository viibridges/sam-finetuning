import numpy as np
import torch
import cv2

if __name__ == '__main__':
    import sys
    sys.path.append('.')

from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import Dataset

import mtutils as mt

class Preprocessor(object):
    def __init__(self, img_size, device):
        super().__init__()
        self.device = device
        self.image_resizer = ResizeLongestSide(img_size)

    def to_tensor(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        x = x.permute(2, 0, 1).contiguous()[:, :, :]
        return x

    def __call__(self, image: np.array) -> torch.Tensor:
        """
        1) Resize image
        2) Normalize pixel values and pad to a square input
        3) Permute channels to CxHxW and convert to tensor
        """
        # resizing
        x = self.image_resizer.apply_image(image)

        # normalization
        x = (x - [[[123.675, 116.28, 103.53]]]) / [[[58.395, 57.12, 57.375]]]

        # to tensor
        x = self.to_tensor(x)

        return x


class JsonDataset(Dataset):
    def __init__(self, json_file, image_data_root, img_size=None, device='cpu'):
        self.data = mt.DataManager.load(json_file)
        self.image_data_root = image_data_root
        self.preprocessor = Preprocessor(img_size, device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        rec = self.data[index]

        # load image
        image_path = mt.osp.join(self.image_data_root, rec['info']['image_path'])
        image = mt.cv_rgb_imread(image_path)

        # load mask
        mask = np.zeros(image.shape[:2], dtype='bool')
        for mpth in rec['info']['mask_path']:
            m = cv2.imread(mt.osp.join(self.image_data_root, mpth), 0) > 0
            mask = np.logical_or(mask, m)

        # preprocess image and bboxes
        image_tensor = self.preprocessor(image)

        # preprocess mask
        mask = (self.preprocessor.image_resizer.apply_image(mask.astype('uint8')) > 0).astype('uint8')
        mask_tensor = self.preprocessor.to_tensor(mask[...,None].astype('float32'))

        return image_tensor, mask_tensor


if __name__ == '__main__':
    image_data_root = '/workspace/dataSet/dataset/sam-finetuning/'
    json_path = mt.osp.join(image_data_root, 'val.json')
    dataset = JsonDataset(json_path, image_data_root, img_size=512)
    for img, msk in dataset:
        mask = np.squeeze(msk.cpu().detach().numpy())
        imag = img.cpu().detach().numpy()

        # mask = (mask*180).astype('uint8')
        mt.PIS(imag[0], imag[1], imag[2], mask)