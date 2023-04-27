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
    def __init__(self, img_size):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_resizer = ResizeLongestSide(img_size)
        self.mask_resizer  = ResizeLongestSide(img_size//4)

    def to_tensor(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        x = x.permute(2, 0, 1).contiguous()[:, :, :]
        return x

    def __call__(self, image: np.array, bboxes: np.array) -> torch.Tensor:
        """
        1) Resize image
        2) Normalize pixel values and pad to a square input
        3) Permute channels to CxHxW and convert to tensor
        """
        # resizing
        x = self.image_resizer.apply_image(image)
        y = self.image_resizer.apply_boxes(bboxes, image.shape[:2])

        # normalization
        x = (x - [[[123.675, 116.28, 103.53]]]) / [[[58.395, 57.12, 57.375]]]

        # to tensor
        x = self.to_tensor(x)
        y = torch.as_tensor(y, dtype=torch.float32, device=self.device)

        return x, y


class JsonDataset(Dataset):
    def __init__(self, json_file, image_data_root, img_size=None):
        self.data = mt.DataManager.load(json_file)
        self.image_data_root = image_data_root

        self.preprocessor = Preprocessor(img_size)

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

        # remove small blocks
        mask = mt.connected_filter(mask, min_area=100)

        # load bboxes
        bboxes = np.array(mt.get_xyxys_from_mask(mask))

        # TODO: only one bboxes are accepted
        if len(bboxes) > 1:
            bboxes = bboxes[:1]
        else:
            bboxes = np.array([[-1,-1,-1,-1]])

        # preprocess image and bboxes
        image_tensor, bboxes_tensor = self.preprocessor(image, bboxes)

        # preprocess mask
        mask = (self.preprocessor.mask_resizer.apply_image(mask.astype('uint8')) > 0).astype('uint8')
        mask_tensor = self.preprocessor.to_tensor(mask[...,None].astype('float32'))

        return image_tensor, mask_tensor, bboxes_tensor


if __name__ == '__main__':
    image_data_root = '/workspace/dataSet/dataset/sam-finetuning/'
    json_path = mt.osp.join(image_data_root, 'val.json')
    dataset = JsonDataset(json_path, image_data_root, img_size=512)
    for img, msk, prmpt in dataset:
        mask = np.squeeze(msk.cpu().detach().numpy())
        imag = img.cpu().detach().numpy()
        bbox = prmpt.cpu().detach().numpy().tolist()

        mask = (mask*180).astype('uint8')
        mask = mt.boxes_painter(mask, bbox, color=(255, 255, 255), line_thickness=1)
        mt.PIS(imag[0], imag[1], imag[2], mask)