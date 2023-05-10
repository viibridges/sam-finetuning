import torch
from segment_anything.custom.datasets import *
from segment_anything.custom.losses import *
from segment_anything import sam_model_registry

from config import cfg

from sklearn.metrics import average_precision_score

def validate_model(model, val_dataloader):
    model.eval()

    aps = list()
    with torch.no_grad():
        for images, gt_masks, _ in mt.tqdm(val_dataloader):
            image_embedding = model.image_encoder(images)
            low_res_mask = model.mask_encoder(image_embedding)
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points = None,
                boxes = None,
                masks = low_res_mask,
                )
            pred_masks, pred_ious = model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                )

            image_size = [model.image_encoder.img_size]*2

            if cfg.sigmoid_out:
                ng_masks = torch.sigmoid(pred_masks[:,0:1])
            else:
                ng_masks = pred_masks[:,0:1]

            ng_masks_predit = model.postprocess_masks(ng_masks, image_size, image_size)
            ng_masks_target = torch.clamp(gt_masks, max=1).unsqueeze(1)

            y_true = ng_masks_target.reshape(-1).cpu().numpy()
            y_pred = ng_masks_predit.reshape(-1).cpu().numpy()
            ap = average_precision_score(y_true, y_pred)

            aps.append(ap)

    model.train()

    return np.mean(aps)


if __name__ == '__main__':
    device = 'cuda:{}'.format(mt.get_single_gpu_id())
    data_test  = JsonDataset(cfg.json_test, cfg.image_data_root, img_size=cfg.image_size, device=device)
    dataloader_test = torch.utils.data.DataLoader(data_test, batch_size=cfg.batch_size, shuffle=False)

    # Set up model
    model = sam_model_registry[cfg.model_type](
        image_size=cfg.image_size, checkpoint=cfg.checkpoint, val=False).to(device)
        # image_size=cfg.image_size, checkpoint=cfg.model_path, val=True).to(device)

    # testing to get AP
    ap = validate_model(model, dataloader_test)

    print("AP in test set: {}".format(ap))