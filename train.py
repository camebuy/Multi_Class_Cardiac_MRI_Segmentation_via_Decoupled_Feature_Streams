import argparse
import os
from pathlib import Path
import random
import time
import datetime
import numpy as np
import albumentations as A
import torch
from torch.utils.data import DataLoader
from utils.utils import print_and_save, shuffling, epoch_time
from network.model import CardiacDecoupledSegNet
from utils.losses import CardiacSegmentationLoss

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
from utils.dataset import load_data, DATASET
from utils.run_engine import train, evaluate


def my_seeding(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train cardiac segmentation with decoupled feature streams.")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    parser.add_argument("--resume", type=Path, default=None, help="Load model weights; optimizer/scheduler restart as before.")
    args = parser.parse_args()

    dataset_name = 'ACDC2017'
    val_name = None

    seed = 0
    my_seeding(seed)

    image_size = 256
    batch_size = 4
    num_epochs = 300
    lr = 2e-4
    lr_backbone = 2e-4
    early_stopping_patience = 100
    weight_decay = 1e-4

    resume_path = args.resume

    note_name = None
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if note_name is not None:
        folder_name = f"{note_name}_{current_time}"
    else:
        folder_name = f"{dataset_name}_{val_name}_lr{lr}_{current_time}"


    # Directories
    base_dir = args.data_root
    data_path = os.path.join(base_dir, dataset_name)
    save_dir = os.path.join("run_files", dataset_name, folder_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 创建一个文本文件用于储存训练信息，创建一个模型文件用于保存训练完成的模型
    train_log_path = os.path.join(save_dir, "train_log.txt")
    checkpoint_path = os.path.join(save_dir, "checkpoint.pth")
    train_log = open(train_log_path, "w")
    train_log.write("\n")
    train_log.close()

    # 获取当前时间并写入文本文件
    datetime_object = str(datetime.datetime.now())
    print_and_save(train_log_path, datetime_object)
    print("")

    # 在文本文件中写入基础超参数信息
    hyperparameters_str = f"Image Size: {image_size}\nBatch Size: {batch_size}\nLR: {lr}\nEpochs: {num_epochs}\n"
    hyperparameters_str += f"Early Stopping Patience: {early_stopping_patience}\n"
    hyperparameters_str += f"Seed: {seed}\n"
    print_and_save(train_log_path, hyperparameters_str)

    """ Data augmentation: Transforms """
    # 用Albumentations库创建图像增强流水线
    transform = A.Compose([
        A.Rotate(limit=35, p=0.4),
        A.HorizontalFlip(p=0.4),
        A.VerticalFlip(p=0.4),
        # 增加：弹性变换，模拟心脏的自然形变，不破坏拓扑结构
        A.ElasticTransform(
            p=0.4,
            alpha=120,
            sigma=120 * 0.05,
            alpha_affine=120 * 0.03
        ),
        # 可选：增加亮度/对比度扰动，模拟不同扫描条件
        A.RandomBrightnessContrast(p=0.3),
    ])

    """ Dataset """
    # 读取数据，并将训练数据打乱，将训练数据和验证数据的数量保存到文本文件
    (train_x, train_y), (valid_x, valid_y) = load_data(data_path, val_name)
    train_x, train_y = shuffling(train_x, train_y)
    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print_and_save(train_log_path, data_str)

    """ Dataset and loader """
    # 各种预处理归一化之类的
    train_dataset = DATASET(train_x, train_y, (image_size, image_size), transform=transform)
    valid_dataset = DATASET(valid_x, valid_y, (image_size, image_size), transform=None)

    # 创建数据加载器
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    """ Model """
    # 加载模型
    device = torch.device('cuda')
    model = CardiacDecoupledSegNet(num_classes=4)  # Model construction entry point.

    if resume_path:
        checkpoint = torch.load(resume_path, map_location='cpu')
        model.load_state_dict(checkpoint)

    model = model.to(device)


    param_groups = [
        {'params': [], 'lr': lr_backbone, 'weight_decay': weight_decay},
        {'params': [], 'lr': lr, 'weight_decay': weight_decay}
    ]

    for name, param in model.named_parameters():
        if name.startswith('layer0') or name.startswith('layer1') or name.startswith('layer2') or name.startswith(
                'layer3'):
            param_groups[0]['params'].append(param)
        else:
            param_groups[1]['params'].append(param)

    assert len(param_groups[0]['params']) > 0, "Layer group is empty!"
    assert len(param_groups[1]['params']) > 0, "Rest group is empty!"

    # 使用 AdamW 优化器（比 Adam 更适合带 weight_decay 的情况）
    optimizer = torch.optim.AdamW(param_groups)

    # 创建学习率调制器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=2e-6)
    # 初始化损失函数
    loss_fn = CardiacSegmentationLoss(num_classes=4).to(device)
    # 将损失函数名称打印保存
    loss_name = "CardiacSegmentationLoss"
    data_str = f"Optimizer: AdamW\nLoss: {loss_name}\n"
    print_and_save(train_log_path, data_str)

    # 打印计算可训练参数总量
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    data_str = f"Number of parameters: {num_params / 1000000}M\n"
    print_and_save(train_log_path, data_str)

    """ Training the model """
    # 提前写上各种损失数据
    with open(os.path.join(save_dir, "train_log.csv"), "w") as f:
        f.write(
            "epoch,"
            "train_RV_Dice,train_LV_Dice,train_MY_Dice,train_AVERAGE_Dice,"
            "train_RV_HD95,train_LV_HD95,train_MY_HD95,train_AVERAGE_HD95,"
            "val_RV_Dice,val_LV_Dice,val_MY_Dice,val_AVERAGE_Dice,"
            "val_RV_HD95,val_LV_HD95,val_MY_HD95,val_AVERAGE_HD95\n")

    # 初始化，为了保存模型和早停
    best_valid_dice  = 0.0
    early_stopping_count = 0

    # 开始训练循环
    for epoch in range(num_epochs):
        # 获取开始时间
        start_time = time.time()

        # 训练，验证
        train_loss, TR_RV_DICE, TR_MY_DICE, TR_LV_DICE, TR_RV_HD, TR_MY_HD, TR_LV_HD = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss, VA_RV_DICE, VA_MY_DICE, VA_LV_DICE, VA_RV_HD, VA_MY_HD, VA_LV_HD = evaluate(model, valid_loader, loss_fn, device)
        # 调整学习率
        scheduler.step()

        TRMeanDice = (TR_RV_DICE + TR_MY_DICE + TR_LV_DICE) / 3
        VAMeanDice = (VA_RV_DICE + VA_MY_DICE + VA_LV_DICE) / 3
        TRMeanHD = (TR_RV_HD + TR_MY_HD + TR_LV_HD) / 3
        VAMeanHD = (VA_RV_HD + VA_MY_HD + VA_LV_HD) / 3

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n"
        data_str += f"\tTrain RV_Dice: {TR_RV_DICE:.4f} - Train LV_Dice: {TR_LV_DICE:.4f} - Train MY_Dice: {TR_MY_DICE:.4f} - Train AVERAGE Dice: {TRMeanDice:.4f}\n"
        data_str += f"\tTrain RV HD  : {TR_RV_HD:.4f} - Train LV HD  : {TR_LV_HD:.4f} - Train MY HD  : {TR_MY_HD:.4f} - Train AVERAGE HD   : {TRMeanHD:.4f}\n"
        data_str += f"\t Val  RV_Dice: {VA_RV_DICE:.4f} -  Val  LV_Dice: {VA_LV_DICE:.4f} -  Val  MY_Dice: {VA_MY_DICE:.4f} -  Val  AVERAGE Dice: {VAMeanDice:.4f}\n"
        data_str += f"\t Val  RV HD  : {VA_RV_HD:.4f} -  Val  LV HD  : {VA_LV_HD:.4f} -  Val  MY HD  : {VA_MY_HD:.4f} -  Val  AVERAGE HD   : {VAMeanHD:.4f}\n"
        print_and_save(train_log_path, data_str)

        # 保存性能最优的模型，并延迟早停触发。
        if VAMeanDice > best_valid_dice:
            data_str = f"Valid DICE improved from {best_valid_dice:2.4f} to {VAMeanDice:2.4f}. Saving checkpoint: {checkpoint_path}"
            print_and_save(train_log_path, data_str)

            best_valid_dice = VAMeanDice
            torch.save(model.state_dict(), checkpoint_path)
            early_stopping_count = 0
        # 早停累积计数
        elif VAMeanDice < best_valid_dice:
            early_stopping_count += 1

        with open(os.path.join(save_dir, "train_log.csv"), "a") as f:
            f.write(
                f"{epoch + 1},{TR_RV_DICE},{TR_LV_DICE},{TR_MY_DICE},{TRMeanDice},"
                f"{TR_RV_HD},{TR_LV_HD},{TR_MY_HD},{TRMeanHD},"
                f"{VA_RV_DICE},{VA_LV_DICE},{VA_MY_DICE},{VAMeanDice},"
                f"{VA_RV_HD},{VA_LV_HD},{VA_MY_HD},{VAMeanHD},\n")

        if early_stopping_count == early_stopping_patience:
            data_str = f"Early stopping: validation mean Dice stops improving from last {early_stopping_patience} continously.\n"
            print_and_save(train_log_path, data_str)
            break

if __name__ == "__main__":
    main()
