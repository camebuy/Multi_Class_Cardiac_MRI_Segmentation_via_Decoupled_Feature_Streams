import torch
import torch.nn as nn


# ==========================================
# 1. 基础组件
# ==========================================
class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False, stride=stride),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act:
            x = self.relu(x)
        return x


# ==========================================
# 2. Multi-class feature decoupling
# ==========================================
class MultiClassFeatureDecoupling(nn.Module):
    def __init__(self, in_c=1024, out_c=128):
        super().__init__()

        # --- A. 前景细分解耦 ---
        self.branch_fg_lv = self._make_main_branch(in_c, out_c)
        self.branch_fg_rv = self._make_main_branch(in_c, out_c)
        self.branch_fg_myo = self._make_main_branch(in_c, out_c)

        # --- B. 背景解耦 ---
        self.branch_bg = self._make_main_branch(in_c, out_c)

        # --- C. 不确定性解耦 (Lite版: 共享主干 + 投影) ---
        self.branch_uc_shared = self._make_main_branch(in_c, out_c)

        # 拓扑边界投影 (1x1 卷积)
        self.proj_uc_lv_myo = nn.Conv2d(out_c, out_c, kernel_size=1, bias=False)
        self.proj_uc_rv_myo = nn.Conv2d(out_c, out_c, kernel_size=1, bias=False)
        self.proj_uc_myo_bg = nn.Conv2d(out_c, out_c, kernel_size=1, bias=False)
        self.proj_uc_rv_bg = nn.Conv2d(out_c, out_c, kernel_size=1, bias=False)

        self.fusion_conv = nn.Conv2d(384, 128, kernel_size=1)

    def _make_main_branch(self, in_c, out_c):
        return nn.Sequential(
            CBR(in_c, 512, kernel_size=3, padding=1),
            CBR(512, 256, kernel_size=3, padding=1),
            CBR(256, out_c, kernel_size=1, padding=0)
        )

    def forward(self, x):
        # 1. 细分前景特征
        f_fg_lv = self.branch_fg_lv(x)
        f_fg_rv = self.branch_fg_rv(x)
        f_fg_myo = self.branch_fg_myo(x)

        # 2. 背景特征
        f_bg = self.branch_bg(x)

        # 3. 细分不确定性特征
        f_uc_shared = self.branch_uc_shared(x)
        f_uc_lv_myo = self.proj_uc_lv_myo(f_uc_shared)
        f_uc_rv_myo = self.proj_uc_rv_myo(f_uc_shared)
        f_uc_myo_bg = self.proj_uc_myo_bg(f_uc_shared)
        f_uc_rv_bg = self.proj_uc_rv_bg(f_uc_shared)

        f_fg_total = torch.cat([f_fg_lv, f_fg_rv, f_fg_myo], dim=1)
        f_fg_total = self.fusion_conv(f_fg_total)  # 增加一个 1x1 卷积层

        return {
            # 这里的 keys 对应原版 model.py 里的返回值解包顺序
            # 或者是给 AuxiliaryHead 用的详细数据
            "main_stream": (f_fg_total, f_bg, f_uc_shared),  # 喂给 CDFA 的
            "details": {
                "fg_subs": (f_fg_lv, f_fg_rv, f_fg_myo),
                "bg": f_bg,
                "uc_subs": (f_uc_lv_myo, f_uc_rv_myo, f_uc_myo_bg, f_uc_rv_bg)
            }
        }


# ==========================================
# 3. Auxiliary semantic and boundary prediction heads
# ==========================================
class AuxiliarySegmentationHead(nn.Module):
    def __init__(self, in_c=128):
        super().__init__()

        # 语义头：改为4通道多分类输出
        self.head_fg_all = nn.Sequential(
            CBR(in_c * 4, 256, kernel_size=3, padding=1),  # 输入4个特征图拼接
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(256, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(64, 32, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(32, 4, kernel_size=1, padding=0),  # 输出4个类别logits
            # 注意：这里不使用激活函数，损失函数中会处理
        )

        # 边界头保持不变（边界区域允许重叠）
        self.head_uc_lv_myo = self._make_uc_head(in_c)
        self.head_uc_rv_myo = self._make_uc_head(in_c)
        self.head_uc_myo_bg = self._make_uc_head(in_c)
        self.head_uc_rv_bg = self._make_uc_head(in_c)

    def _make_uc_head(self, in_c):
        """创建边界预测头（保持Sigmoid二分类）"""
        return nn.Sequential(
            CBR(in_c, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(64, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(64, 32, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(32, 32, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(32, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, details_dict):
        """前向传播，返回多分类logits和边界预测"""
        # 获取各分支特征
        f_fg_lv, f_fg_rv, f_fg_myo = details_dict["fg_subs"]
        f_bg = details_dict["bg"]
        f_uc_lv_myo, f_uc_rv_myo, f_uc_myo_bg, f_uc_rv_bg = details_dict["uc_subs"]

        # 拼接4个特征图
        f_combined = torch.cat([f_fg_lv, f_fg_rv, f_fg_myo, f_bg], dim=1)

        # 多分类语义预测（输出logits，不需要Softmax）
        semantic_logits = self.head_fg_all(f_combined)  # [B, 4, H, W]

        # 边界预测（保持Sigmoid二分类）
        m_uc_lv_myo = self.head_uc_lv_myo(f_uc_lv_myo)
        m_uc_rv_myo = self.head_uc_rv_myo(f_uc_rv_myo)
        m_uc_myo_bg = self.head_uc_myo_bg(f_uc_myo_bg)
        m_uc_rv_bg = self.head_uc_rv_bg(f_uc_rv_bg)

        # 返回：语义logits + 边界预测元组
        return semantic_logits, (m_uc_lv_myo, m_uc_rv_myo, m_uc_myo_bg, m_uc_rv_bg)
