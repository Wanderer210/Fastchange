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

def apply_global_illumination(img_bgr, k, use_linear=True, gamma=2.2):
    """
    img_bgr: uint8 HxWx3
    k: 全局亮度倍率（>1 变亮，<1 变暗）
    use_linear: True 时在“线性光强域”乘 k（更物理）
    """
    if use_linear:
        img = img_bgr.astype(np.float32) / 255.0
        img_lin = srgb_to_linear(img, gamma=gamma)
        img_lin *= float(k)
        out = (linear_to_srgb(img_lin, gamma=gamma) * 255.0 + 0.5).astype(np.uint8)
        return out
    else:
        out = img_bgr.astype(np.float32) * float(k)
        return np.clip(out, 0, 255).astype(np.uint8)

def main():
    # ===== 你只需要改这些参数 =====
    in_dir = Path("Datasets/Car_sorted")            # 或 Path("Fan/sorted")
    out_dir = Path("Datasets/Car_sorted_global")    # 输出目录

    illum_n = 50             # 仅中间 illum_n 帧添加光照
    seed = 123                # 随机种子（影响每张亮度倍率采样）

    # 光照倍率范围（每张“on”图随机采样一个 k）
    k_min, k_max = 1.5, 2.5   # 变亮示例；若想包含变暗可设 0.6~1.8
    use_linear = True         # 推荐 True：线性域乘法更物理

    save_video = True
    video_path = out_dir / "output.mp4"
    video_fps = 30
    # ===========================

    if not in_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {in_dir.resolve()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files = sorted(files, key=natural_key)
    if not files:
        raise FileNotFoundError(f"No images found in: {in_dir.resolve()}")

    rng = np.random.default_rng(seed)

    meta_path = out_dir / "meta.csv"

    illum_start = max(0, (len(files) - int(illum_n)) // 2)
    illum_end = min(len(files), illum_start + int(illum_n))

    illum_k = float(rng.uniform(k_min, k_max))

    if float(video_fps) > 0 and illum_end > illum_start:
        illum_t_start = illum_start / float(video_fps)
        illum_t_end = (illum_end - 1) / float(video_fps)
    else:
        illum_t_start = 0.0
        illum_t_end = 0.0

    vw = None
    video_size = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    try:
        with open(meta_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["index", "filename", "is_on", "k", "illum_start", "illum_end", "illum_n"])

            for i, p in enumerate(files):
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if img is None:
                    raise RuntimeError(f"Failed to read image: {p}")

                is_on = (illum_start <= i < illum_end)

                if is_on:
                    k = illum_k
                    out = apply_global_illumination(img, k, use_linear=use_linear)
                else:
                    k = 1.0
                    out = img

                out_path = out_dir / p.name
                cv2.imwrite(str(out_path), out)

                if save_video:
                    if vw is None:
                        H, W = out.shape[:2]
                        video_size = (W, H)
                        vw = cv2.VideoWriter(str(video_path), fourcc, float(video_fps), video_size)
                        if not vw.isOpened():
                            raise RuntimeError(f"Failed to open video writer: {video_path}")

                    frame = out
                    if (out.shape[1], out.shape[0]) != video_size:
                        frame = cv2.resize(out, video_size, interpolation=cv2.INTER_AREA)
                    vw.write(frame)

                w.writerow([i, p.name, int(is_on), f"{k:.4f}", int(illum_start), int(illum_end), int(illum_n)])

                if (i + 1) % 50 == 0 or (i + 1) == len(files):
                    print(f"[{i+1:4d}/{len(files)}] saved -> {out_path.name}")
    finally:
        if vw is not None:
            vw.release()

    print(f"[OK] Output frames: {out_dir.resolve()}")
    print(f"[OK] Meta saved:    {meta_path.resolve()}")
    if save_video:
        print(f"[OK] Video saved:   {video_path.resolve()}")
    print(f"[OK] Illumination frames: [{int(illum_start)}, {int(illum_end)}) of {len(files)} (n={int(illum_end - illum_start)})")
    print(f"[OK] Illumination time:  start={illum_t_start:.6f}s, end={illum_t_end:.6f}s (fps={float(video_fps):.6f})")

if __name__ == "__main__":
    main()
