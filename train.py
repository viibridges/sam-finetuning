from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
import cv2
import numpy as np
import argparse
import torch
from utils.datasets import *
def parse_args():
    parse=argparse.ArgumentParser(description="mmse test the images or videos")
    parse.add_argument("--model_type",default='vit_b' ,help="vit_b, vit_l, vit_h, ascend in size")
    parse.add_argument("--checkpoint_path",default='model/sam_vit_b_01ec64.pth', help='model/sam_vit_b_01ec64.pth, model/sam_vit_l_0b3195.pth, model/sam_vit_h_4b8939.pth')
    parse.add_argument("--mode",default='train', help='train, val, test')

    args=parse.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    # building model and loading pre-trained checkpoint
    print('building model and loading pre-trained checkpoint')
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint_path)
    
    # create optimizer
    optimizer = torch.optim.AdamW(sam.parameters(),lr=8e-4,betas=(0.9,0.999))  
    
    # create dataloader
    data_root = '/home/dongxinyu/project/sam-finetuning/datasets/'
    imgsz = (512,512)
    batch_size = 2
    train_loader, dataset = create_dataloader(data_root, imgsz, batch_size)    
    for img, label in dataset:
        print(img.shape)
        print(label)

    # TODO: training loop
    