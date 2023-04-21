from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
import cv2
import numpy as np
import argparse
import torch
from utils.datasets import *
from utils.losses import *
from torch.nn.functional import threshold, normalize
import torchvision

import torch.nn as nn

def parse_args():
    parse=argparse.ArgumentParser(description="mmse test the images or videos")
    parse.add_argument("--model_type",default='vit_b' ,help="vit_b, vit_l, vit_h, ascend in size")
    parse.add_argument("--checkpoint_path",default='/home/dongxinyu/project/sam-finetuning/model/sam_vit_b_01ec64.pth', help='model/sam_vit_b_01ec64.pth, model/sam_vit_l_0b3195.pth, model/sam_vit_h_4b8939.pth')
    parse.add_argument("--mode",default='train', help='train, val, test')

    args=parse.parse_args()
    return args

def visual_pred(binary_mask):
    
    # assuming your tensor is named 'tensor'
    image_tensor = binary_mask.squeeze().detach().cpu()  # remove any extra dimensions and move to CPU

    # normalize the tensor to [0, 255] range and convert to PIL image format
    image_tensor = 255 * (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())  # normalize to [0, 255]
    image_tensor = image_tensor.to(torch.uint8)  # convert to uint8 format
    image = torchvision.transforms.functional.to_pil_image(image_tensor, mode='L')  # convert to 8-bit grayscale format

    # save the image to a file
    image.save('my_image.png')

def train_model(model, optimizer, loss_fn, train_dataloader, val_dataloader, num_epochs):
    input_size = (model.image_encoder.img_size,model.image_encoder.img_size)
    ori_shape = (1080,1920)
    
    for epoch in range(num_epochs):
        loss_sum = 0

        print('epoch {}/{}'.format(epoch+1,num_epochs))
        for iter, (images, gt_masks, prompts) in enumerate(train_dataloader):  

            with torch.no_grad():
                image_embedding = model.image_encoder(images)
                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points = None,
                    boxes = prompts,
                    masks = None,
                    )
            pred_masks, pred_ious = model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                ) # pred_masks.shape = torch.Size([1, 1, 256, 256])
            
            post_pred_masks = model.postprocess_masks(pred_masks, input_size, ori_shape).to(device) # upscale to original shape
            visual_pred(post_pred_masks)

            post_pred_masks = normalize(threshold(post_pred_masks, 0.0, 0)).to(device)
            loss = loss_fn(post_pred_masks, gt_masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            # if iter !=0 and iter %2 == 0:
            #     mean_loss = loss_sum/2
            #     print('mean_loss ', mean_loss)
            #     loss_sum = 0
        print('---loss {}---'.format(loss_sum/(iter+1)))
        torch.save(model.state_dict(), 'checkpoints/model_{}.pth'.format(epoch))

if __name__ == '__main__':
    args = parse_args()
    
    # Define image and annotation directories
    # image_dir = '/home/dongxinyu/nfs_243/Disk_4T/wai4/kyle_work/96gaodu/0414/images'
    # annotation_dir = '/home/dongxinyu/nfs_243/Disk_4T/wai4/kyle_work/96gaodu/0414/annotations'
    image_dir = '/home/dongxinyu/project/sam-finetuning/demo/images'
    annotation_dir = '/home/dongxinyu/project/sam-finetuning/demo/annotations'

    
    # Create dataset and dataloader
    maskclass = ['container']
    train_dataset = MyDataset(image_dir, annotation_dir, maskclass=maskclass,img_size = 1024)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataset = MyDataset(image_dir, annotation_dir, maskclass=maskclass,img_size = 1024)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    
    # Set up model
    model = sam_model_registry[args.model_type](checkpoint=args.checkpoint_path,val=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    # create optimizer
    # learning_rate = 8e-4
    learning_rate = 10000
    # optimizer = torch.optim.Adam(model.mask_decoder.parameters(),lr=learning_rate,betas=(0.9,0.999))  
    optimizer = torch.optim.SGD(model.mask_decoder.parameters(),lr=learning_rate)  

    # loss function
    loss_fn = FocalLoss()
    # loss_fn = nn.MSELoss()
    
    # training
    num_epochs = 10
    trained_model = train_model(model, optimizer, loss_fn, train_dataloader, val_dataloader, num_epochs)

    # # Save model
    # torch.save(trained_model, "./model.pt")
