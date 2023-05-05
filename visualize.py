import torch
from segment_anything import sam_model_registry
from segment_anything.custom.datasets import *
from segment_anything.custom.losses import *
from config import cfg
import mtutils as mt

if __name__ == '__main__':
    # setup parameters
    device = 'cuda:{}'.format(mt.get_single_gpu_id())

    data_val  = JsonDataset(cfg.json_val, cfg.image_data_root, img_size=cfg.image_size, device=device)
    dataloader_val = torch.utils.data.DataLoader(data_val, batch_size=1, shuffle=False)

    # Set up model
    model = sam_model_registry[cfg.model_type](
        image_size=cfg.image_size, checkpoint=cfg.model_path, val=True, device=device).to(device)

    for iter, (image, _, raw_image) in enumerate(dataloader_val):  
        with torch.no_grad():
            image_embedding = model.image_encoder(image)

            pred_masks_logits = model.mask_decoder(
                image_embeddings=image_embedding,
                original_size=raw_image.shape[1:3]
            )
            pred_masks = F.softmax(pred_masks_logits, dim=1)
            ng_mask = np.squeeze(pred_masks.detach().cpu().numpy())[1]
            raw_image = np.squeeze(raw_image.detach().cpu().numpy()).astype('uint8')

            mt.PIS(raw_image, ng_mask, norm_float=False, share_xy=True)
