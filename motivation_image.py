import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # 屏蔽 timm 的提示

import mmcv
import torch
import numpy as np
import matplotlib.pyplot as plt
from mmcv import Config, ConfigDict
from mmseg.apis import init_segmentor

# --------------------- 常量：唯一真尺寸 ---------------------
RES_H, RES_W = 128, 256   # 高、宽（注意！mmcv.imresize(size=(W,H))）

# --------------------- 可视化：确保 seg 与 img 尺寸一致 ---------------------
def show_segmentation(img_hwc, seg_pred, palette, opacity=0.6, title=None):
    """
    img_hwc: numpy HWC (H,W,3) BGR
    seg_pred: numpy (H,W) 或 [ (H,W) ]；若尺寸不同会强制拉齐
    """
    # 统一为 HWC 三通道
    if img_hwc.ndim == 2:
        img_hwc = np.stack([img_hwc]*3, axis=-1)
    elif img_hwc.ndim == 3 and img_hwc.shape[0] == 3 and img_hwc.shape[2] != 3:
        # 避免传进来的是 CHW
        img_hwc = np.transpose(img_hwc, (1, 2, 0))

    H, W = img_hwc.shape[:2]

    # 取出 seg 数组
    seg = seg_pred[0] if isinstance(seg_pred, (list, tuple)) else seg_pred
    seg = seg.astype(np.uint8)

    # 尺寸不一致就强制对齐
    if seg.shape != (H, W):
        seg = mmcv.imresize(seg, (W, H), interpolation='nearest')

    # 上色
    seg_color = np.zeros((H, W, 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        mask = (seg == label)            # (H, W)
        if mask.any():
            seg_color[mask] = np.array(color, dtype=np.uint8)

    # 叠加
    blended = (img_hwc * (1 - opacity) + seg_color * opacity).astype(np.uint8)
    plt.imshow(mmcv.bgr2rgb(blended))
    if title:
        plt.title(title, fontsize=12)
    plt.axis("off")

# --------------------- 推理：不在此处 resize！ ---------------------
def prompt_inference_segmentor(model, img_hwc, position=None):
    """
    img_hwc: 已经 resize 好的 HWC (H,W,3) BGR
    返回：numpy (H,W) 的分割结果
    """
    assert isinstance(img_hwc, np.ndarray) and img_hwc.ndim == 3 and img_hwc.shape[2] == 3, \
        f"Expect HWC image, got shape={img_hwc.shape}"
    H, W = img_hwc.shape[:2]

    # 转 tensor (B, C, H, W)
    img_t = torch.from_numpy(img_hwc).float().permute(2, 0, 1).unsqueeze(0).cuda()  # BGR
    img_t = img_t / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_t.device).view(1, 3, 1, 1)  # 用 RGB 的均值没问题用于可视化；若介意，可换 BGR
    std  = torch.tensor([0.229, 0.224, 0.225], device=img_t.device).view(1, 3, 1, 1)
    img_t = (img_t - mean) / std

    img_meta = [{
        "ori_shape": (H, W),
        "img_shape": (H, W),
        "pad_shape": (H, W),
        "filename": None,
        "scale_factor": 1.0,
        "flip": False
    }]

    model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            # encode_decode 会把输出 resize 到 (H, W)
            seg_logit = model.encode_decode(img_t, img_meta, position=position)  # (1, C, H, W)
            seg_pred = seg_logit.argmax(dim=1).squeeze(0).contiguous().cpu().numpy()  # (H, W)

    return seg_pred  # (H, W)

# --------------------- 主程序 ---------------------
if __name__ == "__main__":
    config_file = "local_configs/segformer/B5/segformer.b5.1024x1024.acdc.160k_p.py"
    checkpoint_file = "/home/jonaslei/SVDP/segformer.b5.1024x1024.city.160k.pth"

    cfg = Config.fromfile(config_file)
    cfg.model.test_cfg = ConfigDict(dict(mode="whole"))
    model = init_segmentor(cfg, checkpoint_file, device="cuda:0")

    img_paths = {
        "Cityscapes fog_ref": "/home/jonaslei/SVDP/dataset/rgb_anon/fog/test_ref/GOPR0475/GOPR0475_frame_000260_rgb_ref_anon.png",
        "ACDC fog": "/home/jonaslei/SVDP/dataset/rgb_anon/fog/test/GOPR0475/GOPR0475_frame_000260_rgb_anon.png",
        "Cityscapes night_ref": "/home/jonaslei/SVDP/dataset/rgb_anon/night/test_ref/GOPR0355/GOPR0355_frame_000142_rgb_ref_anon.png",
        "ACDC night": "/home/jonaslei/SVDP/dataset/rgb_anon/night/test/GOPR0355/GOPR0355_frame_000142_rgb_anon.png",
        "Cityscapes snow_ref": "/home/jonaslei/SVDP/dataset/rgb_anon/snow/test_ref/GP030176/GP030176_frame_000129_rgb_ref_anon.png",
        "ACDC snow": "/home/jonaslei/SVDP/dataset/rgb_anon/snow/test/GP030176/GP030176_frame_000129_rgb_anon.png"
    }

    # 假设 img_paths 已经定义
    cols, rows = 6, 2
    fig = plt.figure(figsize=(20, 6))

    cell_w = 1 / cols
    cell_h = 1 / rows

    for i, (domain, img_path) in enumerate(img_paths.items()):
        img = mmcv.imread(img_path)
        img_small = mmcv.imresize(img, (RES_W, RES_H))
        seg_pred = prompt_inference_segmentor(model, img_small, position=None)

        # ---------- 上排 ----------
        ax_top = fig.add_axes([i * cell_w, cell_h, cell_w, cell_h])
        ax_top.imshow(mmcv.bgr2rgb(img_small), aspect='auto')
        ax_top.axis("off")
        ax_top.set_position([i * cell_w, cell_h, cell_w, cell_h])  # 强制位置
        ax_top.set_xlim(0, img_small.shape[1])
        ax_top.set_ylim(img_small.shape[0], 0)  # 防止上下反转
        ax_top.margins(0)
        ax_top.text(5, 15, domain, fontsize=11, color='white',
                    bbox=dict(facecolor='black', alpha=0.5, pad=2, edgecolor='none'))

        # ---------- 下排 ----------
        ax_bot = fig.add_axes([i * cell_w, 0, cell_w, cell_h])
        ax_bot.axis("off")
        ax_bot.set_position([i * cell_w, 0, cell_w, cell_h])
        ax_bot.margins(0)
        plt.sca(ax_bot)
        show_segmentation(img_small, seg_pred, model.PALETTE, opacity=0.6, title=None)

    # ---------- 关键：消除所有残余边距 ----------
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.margins(0)
    fig.patch.set_facecolor('black')  # 避免白缝（用背景色填充）
    plt.savefig("segmentation_comparison_safe.png", dpi=200, bbox_inches="tight", pad_inches=0, facecolor='black')
    plt.show()




