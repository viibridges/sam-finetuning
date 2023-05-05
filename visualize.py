from segment_anything import sam_model_registry
import torch
from segment_anything.custom.datasets import *
from segment_anything.custom.losses import *
import mtutils as mt

if __name__ == '__main__':
    # setup parameters
    image_data_root = '/workspace/dataSet/dataset/sam-finetuning/'
    json_val   = mt.osp.join(image_data_root, 'val.json')
    device = 'cuda:{}'.format(mt.get_single_gpu_id())

    image_size = 512

    model_type = 'vit_b'  # vit_b, vit_l, vit_h, ascend in size
    checkpoint = 'tmp/model_latest.pth'

    data_val  = JsonDataset(json_val, image_data_root, img_size=image_size, device=device)
    dataloader_val = torch.utils.data.DataLoader(data_val, batch_size=1, shuffle=False)

    # Set up model
    model = sam_model_registry[model_type](image_size=image_size, checkpoint=checkpoint, val=True, device=device).to(device)

    for iter, (image, _, raw_image) in enumerate(dataloader_val):  
        with torch.no_grad():
            image_embedding = model.image_encoder(image)

            pred_mask_logits = model.mask_decoder(
                image_embeddings=image_embedding,
                original_size=raw_image.shape[1:3]
            )
            pred_mask = torch.sigmoid(pred_mask_logits)
            pred_mask = np.squeeze(pred_mask.detach().cpu().numpy())
            raw_image = np.squeeze(raw_image.detach().cpu().numpy()).astype('uint8')

            mt.PIS(raw_image, pred_mask, norm_float=False, share_xy=True)
