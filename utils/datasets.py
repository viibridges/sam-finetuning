import os
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything.modeling import Sam

class Normalization(nn.Module):
    def __init__(self, img_size):
        super().__init__()

        pixel_mean = [123.675, 116.28, 103.53]
        pixel_std = [58.395, 57.12, 57.375]
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.img_size = img_size
        
    def preprocess(self, x):
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))

        return x

class MyDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, maskclass=None, img_size=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.maskclass = maskclass
        self.images = os.listdir(image_dir)
        self.annotations = os.listdir(annotation_dir)
        self.img_size = img_size
        self.transform = ResizeLongestSide(self.img_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.norm = Normalization(self.img_size).to(self.device)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        annotation_path = os.path.join(self.annotation_dir, self.annotations[index])

        # Load image
        ori_image = cv2.imread(image_path)
        image = ori_image[..., ::-1] # bgr 2 rgb

        image = self.transform.apply_image(image)
        image = torch.as_tensor(image, device=self.device)
        image = image.permute(2, 0, 1).contiguous()[:, :, :]
        image = self.norm.preprocess(image)

        # Load annotation
        with open(annotation_path, 'r') as f:
            data = json.load(f)
        
        # # Get image size
        width, height = ori_image.shape[1], ori_image.shape[0]
        # Create segmentation mask
        mask = np.zeros((height, width), dtype=np.uint8)

        for shape in data['shapes']:
            label = shape['label']
            if label in self.maskclass:
                points = shape['points']
                polygon = np.array(points, dtype=np.int32)
                cv2.fillPoly(mask, [polygon], color=1)

        # # Convert the binary mask to a NumPy array
        # mask_np = np.array(mask)*255
        # # Save the mask as a JPEG file using OpenCV
        # cv2.imwrite('mask.jpg', mask_np)
        # print('mask saved')

        # mask to tensor
        mask = torch.as_tensor(mask, device=self.device).float()
        mask = mask[None, :, :] # torch.Size([1, 1, 1080, 1920])
        
        # # a box
        # box_prompt = np.array(data['box_prompt'][0], dtype=np.int32)
        # box_prompt = torch.as_tensor(box_prompt, device=self.device)
        # print(prompts.shape)
        # a box
        box_prompt = np.array(data['box_prompt'][0], dtype=np.int32)
        box_prompt = self.transform.apply_boxes(box_prompt, (height, width))
        box_prompt = torch.as_tensor(box_prompt, dtype=torch.float, device=self.device).reshape(4)
        return image, mask, box_prompt
   
if __name__ == '__main__':
    # Define image and annotation directories
    image_dir = '/home/dongxinyu/project/sam-finetuning/demo/images'
    annotation_dir = '/home/dongxinyu/project/sam-finetuning/demo/annotations'

    # Create dataset and dataloader
    maskclass = ['container']
    dataset = MyDataset(image_dir, annotation_dir,maskclass=maskclass,img_size = 1024)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Iterate over the DataLoader to get batches of data
    for batch in dataloader:
        print(batch)