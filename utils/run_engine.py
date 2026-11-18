"""Training and validation loops; metric aggregation is kept unchanged."""

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .metrics import Dice, HD95


def train(model, loader, optimizer, loss_fn, device):
    """
    ACDC 多分类训练函数
    """
    model.train()
    epoch_loss = 0.0
    RV_DICE = 0.0
    MY_DICE = 0.0
    LV_DICE = 0.0
    RV_HD = 0.0
    MY_HD = 0.0
    LV_HD = 0.0

    # 进度条
    loop = tqdm(loader, desc="Training", leave=True)

    for batch_idx, (data, targets) in enumerate(loop):
        # 1. 数据搬运
        data = data.to(device, dtype=torch.float32)


        targets = targets.to(device, dtype=torch.long)

        optimizer.zero_grad()

        # 3. 模型前向 (返回 Tuple)
        # main_logits: [B, 4, H, W] (主干输出)
        # aux_preds: 解耦模块的辅助输出 (语义logits, 边界预测)
        main_preds, aux_preds = model(data)

        # 4. 计算损失
        # loss_fn 是 CardiacSegmentationLoss，它会自动处理 main_logits 和 aux_preds
        loss, loss_details = loss_fn(main_preds, aux_preds, targets)

        # 5. 反向传播
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # 更新进度条显示的 Loss
        loop.set_postfix(
            loss=loss.item(),
            main=loss_details.get("main_ce", None) + loss_details.get("main_dice", None),
            sid=loss_details.get("sid", None)
        )

        probs = F.softmax(main_preds, dim=1)
        Tpreds = torch.argmax(probs, dim=1)

        RV_DICE += Dice(Tpreds, targets, 1)
        MY_DICE += Dice(Tpreds, targets, 2)
        LV_DICE += Dice(Tpreds, targets, 3)
        RV_HD += HD95(Tpreds, targets, 1)
        MY_HD += HD95(Tpreds, targets, 2)
        LV_HD += HD95(Tpreds, targets, 3)


    return epoch_loss / len(loader), RV_DICE / len(loader), MY_DICE / len(loader), LV_DICE / len(loader), RV_HD / len(loader), MY_HD / len(loader), LV_HD / len(loader)


def evaluate(model, loader, loss_fn, device):
    """
    ACDC 验证函数
    """
    model.eval()
    epoch_loss = 0.0
    RV_DICE = 0.0
    MY_DICE = 0.0
    LV_DICE = 0.0
    RV_HD = 0.0
    MY_HD = 0.0
    LV_HD = 0.0


    loop = tqdm(loader, desc="Validating", leave=True)

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device, dtype=torch.float32)
            if targets.ndim == 4:
                targets = targets.squeeze(1)
            targets = targets.to(device, dtype=torch.long)

            # 前向
            main_logits, aux_preds = model(data)

            # 计算验证集 Loss
            loss, _ = loss_fn(main_logits, aux_preds, targets)
            epoch_loss += loss.item()

            # --- Mask 后处理 (用于计算指标) ---
            # 1. Softmax 得到概率图 [B, 4, H, W]
            probs = F.softmax(main_logits, dim=1)

            # 2. Argmax 拿到最终类别预测图 [B, H, W]，值为 0,1,2,3
            Tpreds = torch.argmax(probs, dim=1)


            loop.set_postfix(val_loss=loss.item())

            RV_DICE += Dice(Tpreds, targets, 1)
            MY_DICE += Dice(Tpreds, targets, 2)
            LV_DICE += Dice(Tpreds, targets, 3)
            RV_HD += HD95(Tpreds, targets, 1)
            MY_HD += HD95(Tpreds, targets, 2)
            LV_HD += HD95(Tpreds, targets, 3)

    return epoch_loss / len(loader), RV_DICE / len(loader), MY_DICE / len(loader), LV_DICE / len(loader), RV_HD / len(loader), MY_HD / len(loader), LV_HD / len(loader)
