import torch
from segment_anything import sam_model_registry
from segment_anything.custom.datasets import *
from segment_anything.custom.losses import *
from config import cfg
import mtutils as mt

if __name__ == '__main__':
    # setup parameters
    device = 'cuda:{}'.format(mt.get_single_gpu_id())

    data_test = JsonDataset(cfg.json_test, cfg.image_data_root, img_size=cfg.image_size, device=device)
    dataloader_val = torch.utils.data.DataLoader(data_test, batch_size=1, shuffle=True)

    # Set up model
    model = sam_model_registry[cfg.model_type](
        image_size=cfg.image_size, checkpoint=cfg.model_path, val=True, device=device).to(device)

    for iter, (image, _, raw_image) in enumerate(dataloader_val):  
        with torch.no_grad():
            image_embedding = model.image_encoder(image)
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points = None,
                boxes  = None,
                masks  = None,
                )
            pred_masks, pred_ious = model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                )

            image_size = [model.image_encoder.img_size]*2

            ng_masks = torch.sigmoid(pred_masks)
            ng_masks_predit = model.postprocess_masks(ng_masks, image_size, raw_image.shape[1:3])

            ng_mask = np.squeeze(ng_masks_predit.detach().cpu().numpy())
            raw_image = np.squeeze(raw_image.detach().cpu().numpy()).astype('uint8')

            mt.PIS(raw_image, ng_mask, norm_float=False, share_xy=True)
