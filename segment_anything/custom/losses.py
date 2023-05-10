import torch

def dice_loss(y_pred, y_true):
    smooth = 1e-5
    intersection = (y_pred * y_true).sum(dim=[2, 3])
    y_pred_area = y_pred.sum(dim=[2, 3])
    y_true_area = y_true.sum(dim=[2, 3])
    dice_score = (2 * intersection + smooth) / (y_pred_area + y_true_area + smooth)
    return 1 - dice_score.mean()


def focal_dice_loss(y_pred, y_true, alpha=0.5, gamma=2):
    y_pred = torch.sigmoid(y_pred)
    dice = dice_loss(y_pred, y_true)
    pt = y_pred*y_true + (1-y_pred)*(1-y_true)
    focal_factor = alpha*(1-pt)**gamma
    loss = dice*focal_factor
    return loss.mean()
