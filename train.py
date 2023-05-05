from segment_anything import sam_model_registry
import torch
from utils.datasets import *
from utils.losses import *


def train_model(model, optimizer, loss_fn, train_dataloader, val_dataloader, num_epochs):

    for epoch in range(num_epochs):

        print('epoch {}/{}'.format(epoch+1,num_epochs))

        for iter, (images, gt_masks) in enumerate(train_dataloader):  

            with torch.no_grad():
                image_embedding = model.image_encoder(images)

            pred_masks_logits = model.mask_decoder(
                image_embeddings=image_embedding,
                original_size=images.shape[2:]
            )

            loss = loss_fn(pred_masks_logits, gt_masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('[{}/{}] loss: {}'.format(iter, len(train_dataloader), loss.item()))

        # torch.save(model.state_dict(), 'model_epoch_{}.pth'.format(epoch))
        torch.save(model.state_dict(), 'model_latest.pth')


if __name__ == '__main__':
    # setup parameters
    image_data_root = '/workspace/dataSet/dataset/sam-finetuning/'
    json_train = mt.osp.join(image_data_root, 'train.json')
    json_val   = mt.osp.join(image_data_root, 'val.json')

    device = 'cuda:{}'.format(mt.get_single_gpu_id())

    image_size = 512
    batch_size = 64

    model_type = 'vit_b'  # vit_b, vit_l, vit_h, ascend in size
    checkpoint = mt.osp.join(image_data_root, 'checkpoints/sam_vit_b_01ec64.pth')

    data_train = JsonDataset(json_train, image_data_root, img_size=image_size, device=device)
    dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
    data_val  = JsonDataset(json_train, image_data_root, img_size=image_size, device=device)
    dataloader_val = torch.utils.data.DataLoader(data_val, batch_size=batch_size, shuffle=False)

    # Set up model
    model = sam_model_registry[model_type](image_size=image_size, checkpoint=checkpoint, val=False).to(device)

    # create optimizer
    optimizer = torch.optim.Adam(model.mask_decoder.parameters(),lr=1e-3)

    # loss function
    loss_fn = FocalLoss()
    # loss_fn = nn.MSELoss()
    
    # training
    num_epochs = 10
    trained_model = train_model(model, optimizer, loss_fn, dataloader_train, dataloader_val, num_epochs)

    # # Save model
    # torch.save(trained_model, "./model.pt")
