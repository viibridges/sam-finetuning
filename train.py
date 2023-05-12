import torch
from segment_anything.custom.datasets import *
from segment_anything.custom.losses import *
from segment_anything import sam_model_registry

from test import validate_model
from config import cfg


def train_model(model, train_dataloader, val_dataloader, cfg):
    model.train()

    # create optimizer
    optimizer = torch.optim.Adam(model.mask_decoder.parameters(), lr=1e-3)

    best_metric = -np.Inf
    for epoch in range(cfg.num_epochs):
        ## training loop
        for iter, (images, gt_masks, _) in enumerate(train_dataloader):  
            with torch.no_grad():
                image_embedding = model.image_encoder(images)
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

            ng_masks_predit = model.postprocess_masks(ng_masks, image_size, image_size)
            ng_masks_target = torch.clamp(gt_masks, max=1).unsqueeze(1)
            loss = focal_dice_loss(ng_masks_predit, ng_masks_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Epoch [{}/{}] Iter [{}/{}] loss: {}'.format(epoch+1, cfg.num_epochs,  iter, len(train_dataloader), loss.item()))

        ## carriy out validation
        metric = validate_model(model, val_dataloader)
        print('epoch {}/{}\tValidation AP: {}'.format(epoch+1, cfg.num_epochs, metric))

        ## save model
        if metric > best_metric:
            best_metric = metric
            mt.os.makedirs(cfg.work_dir, exist_ok=True)
            torch.save(model.state_dict(), cfg.model_path)


if __name__ == '__main__':
    device = 'cuda:{}'.format(mt.get_single_gpu_id())

    data_train = JsonDataset(cfg.json_train, cfg.image_data_root, img_size=cfg.image_size, device=device)
    dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=cfg.batch_size, shuffle=True)
    data_val  = JsonDataset(cfg.json_val, cfg.image_data_root, img_size=cfg.image_size, device=device)
    dataloader_val = torch.utils.data.DataLoader(data_val, batch_size=cfg.batch_size, shuffle=False)

    # Set up model
    model = sam_model_registry[cfg.model_type](image_size=cfg.image_size, checkpoint=cfg.checkpoint, val=False).to(device)

    # training
    trained_model = train_model(model, dataloader_train, dataloader_val, cfg)
