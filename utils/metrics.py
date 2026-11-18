"""Historical Dice and HD95 definitions, unchanged for result comparability."""

import numpy as np
import torch


def Dice(pred, target, T_class):
    smooth = 1e-15
    num = pred.size(0)
    A_binary = (pred == T_class).float()
    B_binary = (target == T_class).float()
    A = A_binary.view(num, -1)
    B = B_binary.view(num, -1)
    intersection = (A * B).sum()
    return ((2. * intersection + smooth) / (A.sum() + B.sum() + smooth)).item()


def HD95(pred, target, T_class):
    if torch.is_tensor(pred):
        pred_np = pred.cpu().numpy()
    else:
        pred_np = pred

    if torch.is_tensor(target):
        target_np = target.cpu().numpy()
    else:
        target_np = target

    if pred_np.ndim == 4 and pred_np.shape[1] == 1:
        pred_np = pred_np.squeeze(1)
    if target_np.ndim == 4 and target_np.shape[1] == 1:
        target_np = target_np.squeeze(1)

    batch_size = pred_np.shape[0]
    hd95_values = []

    for i in range(batch_size):
        pred_single = pred_np[i]
        target_single = target_np[i]

        pred_binary = (pred_single == T_class).astype(np.float32)
        target_binary = (target_single == T_class).astype(np.float32)

        pred_points = np.argwhere(pred_binary > 0.5)
        target_points = np.argwhere(target_binary > 0.5)

        if len(pred_points) == 0 and len(target_points) == 0:
            hd95_values.append(0.0)
            continue
        elif len(pred_points) == 0 or len(target_points) == 0:
            height, width = pred_binary.shape
            max_distance = np.sqrt(height ** 2 + width ** 2)
            hd95_values.append(float(max_distance))
            continue

        try:
            from scipy.spatial import cKDTree
            target_tree = cKDTree(target_points)
            distances_pred_to_target, _ = target_tree.query(pred_points, k=1)

            pred_tree = cKDTree(pred_points)
            distances_target_to_pred, _ = pred_tree.query(target_points, k=1)

            all_distances = np.concatenate([distances_pred_to_target, distances_target_to_pred])
            hd95 = np.percentile(all_distances, 95)
            hd95_values.append(float(hd95))
        except ImportError:
            distances_pred_to_target = []
            for p in pred_points:
                min_dist = np.min(np.sqrt(np.sum((p - target_points) ** 2, axis=1)))
                distances_pred_to_target.append(min_dist)

            distances_target_to_pred = []
            for t in target_points:
                min_dist = np.min(np.sqrt(np.sum((t - pred_points) ** 2, axis=1)))
                distances_target_to_pred.append(min_dist)

            all_distances = np.concatenate([distances_pred_to_target, distances_target_to_pred])
            hd95 = np.percentile(all_distances, 95)
            hd95_values.append(float(hd95))

    return np.mean(hd95_values)
