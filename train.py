import torch
from sklearn.metrics import average_precision_score

from segment_anything.custom.datasets import *
from segment_anything.custom.losses import *
from segment_anything import sam_model_registry

from config import cfg


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

def train_model(model, optimizer, loss_fn, train_dataloader, val_dataloader, num_epochs):
    best_metric = -np.Inf
    for epoch in range(num_epochs):
        ## training loop
        for iter, (images, gt_masks, _) in enumerate(train_dataloader):  
            with torch.no_grad():
                image_embedding = model.image_encoder(images)
            pred_masks_logits = model.mask_decoder(
                image_embeddings=image_embedding,
                original_size=images.shape[2:]
            )
            pred_masks = F.softmax(pred_masks_logits, dim=1)

            loss = loss_fn(pred_masks, gt_masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iter%10 == 0:
                print('Epoch [{}/{}] Iter [{}/{}] loss: {}'.format(epoch+1, num_epochs,  iter, len(train_dataloader), loss.item()))

        ## carriy out validation
        metric = validate_model(model, val_dataloader)
        print('epoch {}/{}\tValidation AP: {}'.format(epoch+1, num_epochs, metric))

        ## save model
        if metric > best_metric:
            best_metric = metric
            mt.os.makedirs('tmp/', exist_ok=True)
            torch.save(model.state_dict(), 'tmp/model_latest.pth')


if __name__ == '__main__':
    device = 'cuda:{}'.format(mt.get_single_gpu_id())

    data_train = JsonDataset(cfg.json_train, cfg.image_data_root, img_size=cfg.image_size, device=device)
    dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=cfg.batch_size, shuffle=True)
    data_val  = JsonDataset(cfg.json_val, cfg.image_data_root, img_size=cfg.image_size, device=device)
    dataloader_val = torch.utils.data.DataLoader(data_val, batch_size=cfg.batch_size, shuffle=False)

    # Set up model
    model = sam_model_registry[cfg.model_type](image_size=cfg.image_size, checkpoint=cfg.checkpoint, val=False).to(device)

    # create optimizer
    optimizer = torch.optim.Adam(model.mask_decoder.parameters(), lr=1e-3)

    # loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # training
    trained_model = train_model(model, optimizer, loss_fn, dataloader_train, dataloader_val, cfg.num_epochs)
