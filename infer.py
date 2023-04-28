from segment_anything import sam_model_registry
import torch
from utils.datasets import *
from utils.losses import *

if __name__ == '__main__':
    # setup parameters
    image_data_root = '/workspace/dataSet/dataset/sam-finetuning/'
    json_val   = mt.osp.join(image_data_root, 'val.json')

    image_size = 256

    model_type = 'vit_b'  # vit_b, vit_l, vit_h, ascend in size
    checkpoint = 'model_epoch_1.pth'

    data_val  = JsonDataset(json_val, image_data_root, img_size=image_size)
    dataloader_val = torch.utils.data.DataLoader(data_val, batch_size=1, shuffle=False)

    # Set up model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = sam_model_registry[model_type](image_size=image_size, checkpoint=checkpoint, val=True).to(device)

    for iter, (images, gt_masks, prompts) in enumerate(dataloader_val):  
        with torch.no_grad():
            image_embedding = model.image_encoder(images)

            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points = None,
                boxes = prompts,
                masks = None,
                )

            pred_masks_logits, pred_ious = model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
            )
            pred_masks = torch.sigmoid(pred_masks_logits)

            maxid = np.argmax(np.squeeze(pred_ious.detach().cpu().numpy()))
            masks = pred_masks[0].detach().cpu().numpy()
            mask = masks[maxid]
            mt.PIS(mask, norm_float=False)
