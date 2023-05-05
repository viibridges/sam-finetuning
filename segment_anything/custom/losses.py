import torch.nn as nn
import torch.nn.functional as F
import torch

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, smooth=1.):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        intersection = (inputs * targets).sum()
        dice_coef = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        dice_loss = 1. - dice_coef
        
        return torch.tensor(focal_loss.mean()*20 + dice_loss.mean()).clone().detach().requires_grad_(True)


import torch.nn as nn
import torch

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, input_tensor, target_tensor):

        # calculate the cross entropy
        cross_entropy = nn.functional.binary_cross_entropy_with_logits(input_tensor, target_tensor, reduction='none')

        # calculate the focal loss weights
        pt = torch.exp(-cross_entropy)
        focal_weights = (1 - pt) ** self.gamma

        # calculate the final loss
        loss = (focal_weights * cross_entropy).mean()

        return loss
