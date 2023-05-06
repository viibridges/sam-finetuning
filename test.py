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
        for images, masks, _ in mt.tqdm(val_dataloader):
            image_embedding = model.image_encoder(images)
            logits = model.mask_decoder(
                image_embeddings=image_embedding,
                original_size=images.shape[2:]
            )
            preds = F.softmax(logits, dim=1)
            preds = preds[:,1:2,:,:]  # get the fg mask

            y_true = masks.reshape(-1).cpu().numpy()
            y_pred = preds.reshape(-1).cpu().numpy()
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
        image_size=cfg.image_size, checkpoint=cfg.model_path, val=True).to(device)

    # testing to get AP
    ap = validate_model(model, dataloader_test)

    print("AP in test set: {}".format(ap))