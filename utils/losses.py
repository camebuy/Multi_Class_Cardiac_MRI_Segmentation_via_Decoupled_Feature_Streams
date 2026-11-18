"""Main segmentation loss and auxiliary boundary/area-weighted supervision."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiClassDiceLoss(nn.Module):
    def __init__(self, num_classes=4, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets.long(), num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class DecoupledAuxiliaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()

        # Boundary heads already apply sigmoid; retain probability-based BCE.
        self.bce_loss = nn.BCELoss()

    def get_interface_boundary(self, mask_a, mask_b):
        """ 在线提取精准交界线 """
        dilated_a = F.max_pool2d(mask_a, kernel_size=3, stride=1, padding=1)
        dilated_b = F.max_pool2d(mask_b, kernel_size=3, stride=1, padding=1)
        return ((dilated_a * mask_b) + (dilated_b * mask_a) > 0).float()

    def forward(self, aux_outputs, targets):
        semantic_logits, boundary_preds = aux_outputs

        # 1. 基础辅助语义损失
        loss_semantic = self.ce_loss(semantic_logits, targets.long())

        # 2. 并行计算 4 个类别的单通道 Dice Loss
        probs = F.softmax(semantic_logits, dim=1)  # [B, 4, H, W]
        gt_one_hot = F.one_hot(targets.long(), num_classes=4).permute(0, 3, 1, 2).float()  # [B, 4, H, W]

        # 标准单通道 Dice 矩阵化计算
        inter = (probs * gt_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + gt_one_hot.sum(dim=(2, 3))
        dice_per_class = 1.0 - (2. * inter + 1.0) / (union + 1.0)  # [B, 4]

        # Dynamic area weighting with a bounded weight.
        # 计算真实标签中各类别面积占比
        area_ratio = gt_one_hot.sum(dim=(2, 3)) / (gt_one_hot.shape[2] * gt_one_hot.shape[3])  # [B, 4]
        dynamic_beta = 1.0 / (torch.tanh(area_ratio) + 1e-15)  # [B, 4]

        # Clamp weights for classes absent from a slice.
        dynamic_beta = torch.clamp(dynamic_beta, max=10.0)

        # 动态加权后的独立强化损失
        loss_components_total = (dice_per_class * dynamic_beta).mean()

        # 4. 边界提取与配对计算
        gt_masks = [gt_one_hot[:, i:i + 1, :, :] for i in range(4)]  # 0:BG, 1:RV, 2:Myo, 3:LV
        gt_bounds = [
            self.get_interface_boundary(gt_masks[3], gt_masks[2]),  # LV-Myo
            self.get_interface_boundary(gt_masks[1], gt_masks[2]),  # RV-Myo
            self.get_interface_boundary(gt_masks[2], gt_masks[0]),  # Myo-BG
            self.get_interface_boundary(gt_masks[1], gt_masks[0])  # RV-BG
        ]

        # Sum BCE over the four predicted interfaces.
        loss_boundary = sum(self.bce_loss(pred, gt) for pred, gt in zip(boundary_preds, gt_bounds))

        # 5. 总损失拼接 (平衡公式间数量级)
        return loss_semantic + 0.4 * loss_boundary + 0.5 * loss_components_total


class CardiacSegmentationLoss(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = MultiClassDiceLoss(num_classes=num_classes)
        self.sid_loss_fn = DecoupledAuxiliaryLoss()

    def forward(self, main_pred, aux_preds, targets):
        loss_ce = self.ce_loss(main_pred, targets.long())
        loss_dice = self.dice_loss(main_pred, targets.long())
        loss_main = 0.5 * loss_ce + 0.5 * loss_dice

        # 辅助分支总损失
        loss_sid = self.sid_loss_fn(aux_preds, targets)

        # Combine the main loss and auxiliary loss with the existing 0.1 weight.
        return loss_main + 0.1 * loss_sid, {
            "main_ce": loss_ce.item(),
            "main_dice": loss_dice.item(),
            "sid": loss_sid.item()
        }
