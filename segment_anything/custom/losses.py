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


def iou_loss(pred, target):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3)) - inter
    iou = 1 - (inter / union)

    return iou.mean()


class BBCEWithLogitLoss(torch.nn.Module):
    '''
    Balanced BCEWithLogitLoss
    '''
    def __init__(self):
        super(BBCEWithLogitLoss, self).__init__()

    def forward(self, pred, gt):
        eps = 1e-10
        count_pos = torch.sum(gt) + eps
        count_neg = torch.sum(1. - gt)
        ratio = count_neg / count_pos
        w_neg = count_pos / (count_pos + count_neg)

        bce1 = torch.nn.BCEWithLogitsLoss(pos_weight=ratio)
        loss = w_neg * bce1(pred, gt)

        return loss

bbce_loss = BBCEWithLogitLoss()

def mixed_loss(pred, target):
    return bbce_loss(pred, target) + iou_loss(pred, target)