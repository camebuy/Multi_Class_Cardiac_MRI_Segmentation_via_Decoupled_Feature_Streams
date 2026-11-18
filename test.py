import argparse
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import cv2  # 导入 cv2
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
# ---------- 项目内部模块 ----------
from network.model import CardiacDecoupledSegNet
from utils.dataset import DATASET, load_names  # 复用训练的数据加载逻辑
from utils.metrics import Dice, HD95  # 复用训练指标
from utils.utils import create_dir, seeding
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


# 图像保存开关：True 保存，False 不保存；指标始终按原逻辑保存。
SAVE_PRED_MASKS = True       # 预测掩码图，保存到 pred_masks/
SAVE_OVERLAY_IMAGES = True   # 原图与预测叠加图，保存到 overlay/

def evaluate_multi_class(model, test_loader, device, save_dir, save_preds=True, save_overlay=False):
    """
    全功能测试评估函数
    - 计算 Dice 和 HD95（每个类别）
    - 保存预测掩码（灰度 PNG，像素 0-3）
    - 可选保存原图 + 预测叠加图
    """
    model.eval()

    # 指标累加器
    total_rv_dice = 0.0
    total_lv_dice = 0.0
    total_my_dice = 0.0
    total_rv_hd = 0.0
    total_lv_hd = 0.0
    total_my_hd = 0.0

    # 创建保存目录
    if save_preds:
        pred_mask_dir = os.path.join(save_dir, "pred_masks")
        create_dir(pred_mask_dir)
    if save_overlay:
        overlay_dir = os.path.join(save_dir, "overlay")
        create_dir(overlay_dir)

    with torch.no_grad():
        loop = tqdm(test_loader, desc="Testing")
        for batch_idx, (images, masks) in enumerate(loop):
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.long)  # [B, H, W], 值 0-3

            # ---------- 模型推理 ----------
            main_logits, _ = model(images)  # [B, 4, H, W]
            probs = F.softmax(main_logits, dim=1)
            preds = torch.argmax(probs, dim=1)  # [B, H, W], 值 0-3

            # ---------- 保存预测掩码 ----------
            if save_preds:
                for i in range(images.size(0)):
                    pred_np = preds[i].cpu().numpy().astype(np.uint8)
                    filename = f"pred_{batch_idx * test_loader.batch_size + i:06d}.png"
                    pred_pil = Image.fromarray(pred_np, mode='P')
                    pred_pil.putpalette([0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255])  # 黑,红,绿,蓝
                    pred_pil.save(os.path.join(pred_mask_dir, filename))

            # ---------- 保存原图+预测叠加----------
            if save_overlay:
                for i in range(images.size(0)):
                    # 原图（归一化 [0,1]）转为 0-255 RGB
                    img_np = images[i].cpu().numpy().transpose(1, 2, 0)  # [H,W,C]
                    img_np = (img_np * 255).astype(np.uint8).copy()

                    pred_np = preds[i].cpu().numpy().astype(np.uint8)  # 预测标签

                    # 生成一个空白画布作为最终显示图（先复制原图）
                    display_img = img_np.copy()

                    # 为每个像素生成颜色
                    # 规则：
                    # Use the original class palette and blending behavior.
                    alpha = 0.5  # 透明度

                    # 遍历所有像素（也可以用向量化方式，但为了清晰，用循环，速度影响不大，因为 256x256 x batch=8）
                    # 为了提高速度，可以使用 numpy 向量化操作
                    h, w = pred_np.shape
                    # 创建一个颜色图层（初始为全透明）
                    color_layer = np.zeros((h, w, 3), dtype=np.uint8)
                    color_layer[pred_np == 1] = [255, 0, 0]  # 只要模型预测是 RV，就涂红色
                    color_layer[pred_np == 2] = [0, 255, 0]  # 只要模型预测是 Myo，就涂绿色
                    color_layer[pred_np == 3] = [0, 0, 255]
                    # 将颜色图层以 alpha 透明度与原图叠加
                    blended = cv2.addWeighted(display_img, 1 - alpha, color_layer, alpha, 0)

                    # 保存
                    filename = f"overlay_{batch_idx * test_loader.batch_size + i:06d}.png"
                    cv2.imwrite(os.path.join(overlay_dir, filename), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

            # ---------- 逐样本累加指标 ----------
            for i in range(images.size(0)):
                pred = preds[i].unsqueeze(0)  # [1, H, W]
                mask = masks[i].unsqueeze(0)  # [1, H, W]
                total_rv_dice += Dice(pred, mask, 1)
                total_lv_dice += Dice(pred, mask, 3)
                total_my_dice += Dice(pred, mask, 2)
                total_rv_hd += HD95(pred, mask, 1)
                total_lv_hd += HD95(pred, mask, 3)
                total_my_hd += HD95(pred, mask, 2)

    # ---------- 计算平均值 ----------
    num_samples = len(test_loader.dataset)
    avg_rv_dice = total_rv_dice / num_samples
    avg_lv_dice = total_lv_dice / num_samples
    avg_my_dice = total_my_dice / num_samples
    avg_dice = (avg_rv_dice + avg_lv_dice + avg_my_dice) / 3
    avg_rv_hd = total_rv_hd / num_samples
    avg_lv_hd = total_lv_hd / num_samples
    avg_my_hd = total_my_hd / num_samples
    avg_hd = (avg_rv_hd + avg_lv_hd + avg_my_hd) / 3

    return {
        "RV Dice": avg_rv_dice, "LV Dice": avg_lv_dice, "Myo Dice": avg_my_dice, "Avg Dice": avg_dice,
        "RV HD95": avg_rv_hd, "LV HD95": avg_lv_hd, "Myo HD95": avg_my_hd, "Avg HD95": avg_hd
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate the cardiac segmentation model.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a saved model.state_dict().")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    parser.add_argument("--dataset", default="ACDC2017test")
    parser.add_argument("--name", default="Ours", help="Result subfolder name.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional first-N smoke test; omit for full evaluation.")
    parser.add_argument("--no-save-preds", action="store_true")
    parser.add_argument("--no-save-overlay", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or (args.max_samples is not None and args.max_samples < 1):
        parser.error("batch-size/max-samples must be positive; workers must be nonnegative.")
    # ========== 配置参数 ==========
    seeding(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------- 路径设置（请根据实际情况修改）----------
    dataset_name = args.dataset  # 数据集文件夹名
    now_name = args.name
    checkpoint_path = args.checkpoint
    test_txt = "test.txt"  # 测试集文件名（官方标准）
    size = (256, 256)  # 输入尺寸，必须与训练一致
    batch_size = args.batch_size  # 可根据显存调整
    save_results = True  # 是否保存结果
    save_preds = SAVE_PRED_MASKS and not args.no_save_preds  # 是否保存预测掩码
    save_overlay = SAVE_OVERLAY_IMAGES and not args.no_save_overlay  # 是否保存叠加图（需要时可开启）

    # ---------- 数据根目录 ----------
    data_path = os.path.join(args.data_root, dataset_name)

    # ========== 1. 加载模型 ==========
    model = CardiacDecoupledSegNet(num_classes=4).to(device)  # Model construction entry point.
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    print(f"✅ Model loaded from {checkpoint_path}")

    # ========== 2. 加载测试集路径 ==========
    test_names_path = os.path.join(data_path, test_txt)
    test_x, test_y = load_names(data_path, test_names_path)
    if args.max_samples is not None:
        test_x, test_y = test_x[:args.max_samples], test_y[:args.max_samples]
        now_name = f"{now_name}_smoke{len(test_x)}"
    if not test_x:
        raise ValueError("No test samples found. Check the manifest and its final newline.")
    print(f"📁 Test samples: {len(test_x)}")

    # ========== 3. 创建 Dataset 和 DataLoader ==========
    test_dataset = DATASET(test_x, test_y, size, transform=None)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    # ========== 4. 创建保存目录 ==========
    save_dir = f"results/{dataset_name}/{now_name}"
    if save_results:
        create_dir(save_dir)

    # ========== 5. 执行评估 ==========
    metrics = evaluate_multi_class(
        model=model,
        test_loader=test_loader,
        device=device,
        save_dir=save_dir,
        save_preds=save_preds,
        save_overlay=save_overlay
    )

    # ========== 6. 打印并保存结果 ==========
    print("\n" + "=" * 50)
    print("🔥 ACDC 2017 Test Set Results 🔥")
    print("=" * 50)
    print(f"RV Dice  : {metrics['RV Dice']:.4f}")
    print(f"LV Dice  : {metrics['LV Dice']:.4f}")
    print(f"Myo Dice : {metrics['Myo Dice']:.4f}")
    print(f"Avg Dice : {metrics['Avg Dice']:.4f}")
    print(f"RV HD95  : {metrics['RV HD95']:.4f}")
    print(f"LV HD95  : {metrics['LV HD95']:.4f}")
    print(f"Myo HD95 : {metrics['Myo HD95']:.4f}")
    print(f"Avg HD95 : {metrics['Avg HD95']:.4f}")
    print("=" * 50)

    if save_results:
        result_path = os.path.join(save_dir, "test_metrics.txt")
        with open(result_path, "w") as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value:.4f}\n")
        print(f"📄 Results saved to {result_path}")
        if save_preds:
            print(f"🖼️  Prediction masks saved to {os.path.join(save_dir, 'pred_masks')}")

if __name__ == "__main__":
    main()
