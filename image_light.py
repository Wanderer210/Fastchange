import re
import csv
from pathlib import Path
import numpy as np
import cv2

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def natural_key(p: Path):
    nums = re.findall(r"\d+", p.stem)
    n = int(nums[-1]) if nums else -1
    return (n, p.name.lower())

def srgb_to_linear(x, gamma=2.2):
    return np.clip(x, 0, 1) ** gamma

def linear_to_srgb(x, gamma=2.2):
    return np.clip(x, 0, 1) ** (1.0 / gamma)

def generate_gaussian_mask(shape, center, sigma):
    """
    生成 2D 高斯分布蒙版，用于模拟局部光源
    shape: 图像的 (H, W)
    center: 光源中心 (cy, cx)
    sigma: 光源扩散范围 (sigma_y, sigma_x)
    """
    H, W = shape[:2]
    Y, X = np.ogrid[:H, :W]
    cy, cx = center
    sy, sx = sigma
    
    # 计算高斯分布，中心值为1，向外衰减到0
    mask = np.exp(-(((Y - cy) ** 2) / (2 * sy ** 2) + ((X - cx) ** 2) / (2 * sx ** 2)))
    # 扩展维度为 (H, W, 1) 以便与 (H, W, 3) 的图像数组广播相乘
    return mask[..., np.newaxis]

def apply_local_illumination(img_bgr, k, center, sigma, use_linear=True, gamma=2.2):
    """
    img_bgr: uint8 HxWx3
    k: 中心最高亮度倍率
    center: 光源中心坐标 (cy, cx)
    sigma: 光源衰减半径 (sy, sx)
    """
    # 1. 生成空间光照蒙版 (0.0 ~ 1.0)
    mask = generate_gaussian_mask(img_bgr.shape, center, sigma)
    
    # 2. 映射为真实的光强倍率图
    # 蒙版为1的地方倍率为k，蒙版为0的地方倍率为1（保持原图）
    k_map = 1.0 + (float(k) - 1.0) * mask
    
    if use_linear:
        # 转换到物理线性空间
        img = img_bgr.astype(np.float32) / 255.0
        img_lin = srgb_to_linear(img, gamma=gamma)
        
        # 施加局部光照
        img_lin *= k_map
        
        # 转回 sRGB 视觉空间
        out = (linear_to_srgb(img_lin, gamma=gamma) * 255.0 + 0.5).astype(np.uint8)
        return out
    else:
        out = img_bgr.astype(np.float32) * k_map
        return np.clip(out, 0, 255).astype(np.uint8)

def process_single_folder(in_dir: Path, out_dir: Path, seed: int):
    """处理单个文件夹的光照和元数据生成逻辑"""
    # ===== 参数配置 =====
    illum_n = 50              # 仅中间 illum_n 帧添加光照
    k_min, k_max = 2.0, 4.0   # 局部高光通常倍率可以设得更高（比如 4.0），以模拟HDR耀斑
    use_linear = True         

    save_video = True
    video_path = out_dir / "output.mp4"
    data_fps = 4000           
    video_fps = 30
    # ====================

    out_dir.mkdir(parents=True, exist_ok=True)

    files = [p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files = sorted(files, key=natural_key)
    if not files:
        print(f"Warning: No images found in {in_dir.resolve()}")
        return

    rng = np.random.default_rng(seed)

    meta_path = out_dir / "meta.csv"
    phase_path = out_dir / "phase_timeline.csv"

    illum_start = max(0, (len(files) - int(illum_n)) // 2)
    illum_end = min(len(files), illum_start + int(illum_n))

    illum_k = float(rng.uniform(k_min, k_max))

    # --- 获取图像尺寸以生成随机光源位置 ---
    sample_img = cv2.imread(str(files[0]))
    if sample_img is None:
        print("Error: Could not read the first image to determine size.")
        return
    H, W = sample_img.shape[:2]
    
    # 随机生成局部光源的中心位置（限制在图像 20%~80% 的区域内）
    cx = int(rng.uniform(0.2 * W, 0.8 * W))
    cy = int(rng.uniform(0.2 * H, 0.8 * H))
    
    # 随机生成光源的扩散半径（假设半径为图像宽度的 15% 到 35%）
    radius = float(rng.uniform(0.15 * W, 0.35 * W))
    sigma = (radius, radius)  # 可以设为不同的值来制造椭圆形光源
    center = (cy, cx)

    fps = float(data_fps)
    total_n = len(files)
    normal1_start, normal1_end = 0, illum_start
    anomaly_start, anomaly_end = illum_start, illum_end
    normal2_start, normal2_end = illum_end, total_n

    def phase_stats(name, start, end):
        n = max(0, end - start)
        t_start = (start / fps) if fps > 0 else 0.0
        t_end = ((end - 1) / fps) if (fps > 0 and n > 0) else t_start
        duration = (n / fps) if fps > 0 else 0.0
        return [name, int(start), int(end), int(n), f"{t_start:.6f}", f"{t_end:.6f}", f"{duration:.6f}"]

    phase_rows = [
        phase_stats("normal_before", normal1_start, normal1_end),
        phase_stats("anomaly", anomaly_start, anomaly_end),
        phase_stats("normal_after", normal2_start, normal2_end),
    ]

    vw = None
    video_size = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    try:
        with open(meta_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            # 增加对局部光源信息的记录
            w.writerow(["index", "filename", "is_on", "k", "center_x", "center_y", "radius", "illum_start", "illum_end", "illum_n"])

            for i, p in enumerate(files):
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if img is None:
                    continue

                is_on = (illum_start <= i < illum_end)

                if is_on:
                    out = apply_local_illumination(img, illum_k, center, sigma, use_linear=use_linear)
                else:
                    out = img

                out_path = out_dir / p.name
                cv2.imwrite(str(out_path), out)

                if save_video:
                    if vw is None:
                        # 确保视频尺寸以第一张图为准
                        video_size = (W, H)
                        vw = cv2.VideoWriter(str(video_path), fourcc, float(video_fps), video_size)
                    
                    frame = out
                    if (out.shape[1], out.shape[0]) != video_size:
                        frame = cv2.resize(out, video_size, interpolation=cv2.INTER_AREA)
                    vw.write(frame)

                # 只有打开光照的时候记录光源信息，否则记为 -1
                record_k = illum_k if is_on else 1.0
                record_cx = cx if is_on else -1
                record_cy = cy if is_on else -1
                record_r = radius if is_on else -1
                
                w.writerow([i, p.name, int(is_on), f"{record_k:.4f}", record_cx, record_cy, f"{record_r:.2f}", int(illum_start), int(illum_end), int(illum_n)])

        with open(phase_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["phase", "start_idx", "end_idx_exclusive", "frame_count", "start_time_s", "end_time_s", "duration_s"])
            w.writerows(phase_rows)

    finally:
        if vw is not None:
            vw.release()

    print(f"[OK] Output frames: {out_dir.resolve()}")
    print(f"[OK] Meta saved:    {meta_path.name} & {phase_path.name}")
    print(f"[OK] Illumination:  k={illum_k:.2f}, center=({cx}, {cy}), radius={radius:.1f}")
    for row in phase_rows:
        print(f"     - {row[0]:13s} frames=[{row[1]}, {row[2]}) count={row[3]}")


def main():
    # ==== 基础路径设置 ====
    BASE_IN_DIR = Path("/home/zhaoyue/Fastchange/Datasets/Scene_sorted")
    BASE_OUT_DIR = Path("/home/zhaoyue/Fastchange/Datasets/Scene-light")
    
    base_seed = 123  # 基础随机种子

    for i in range(1, 11):
        in_dir = BASE_IN_DIR / str(i)
        out_dir = BASE_OUT_DIR / str(i)
        
        if in_dir.exists():
            print(f"\n======================================")
            print(f"  Adding Local HDR Lighting to Folder: {i}  ")
            print(f"======================================")
            process_single_folder(in_dir, out_dir, seed=base_seed + i)
        else:
            print(f"Warning: Input folder not found, skipping: {in_dir}")

if __name__ == "__main__":
    main()