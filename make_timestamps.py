import re
import csv
import shutil
from pathlib import Path

import cv2

FPS = 4000.0
FAN_DIR = Path("Datasets/Fan")  # 改成你的目录（绝对路径或相对路径都行）

EXPORT_SORTED = True  # 是否导出重命名后的有序序列
SORTED_DIR = Path("Datasets/Fan_sorted")   # 导出目录
OUT_CSV = SORTED_DIR / "fan_sequence_timestamps.csv"

SAVE_VIDEO = True   
VIDEO_PATH = SORTED_DIR / "fan_sequence.mp4"
VIDEO_FPS = 30.0
VIDEO_FOURCC = "mp4v"


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# 解析规则：
# 1) A_B.png -> idx = A*4 + B
# 2) A.png   -> idx = A*4 + 3  (即在最后一个时间戳出现)
PAT = re.compile(r"^(?P<a>\d+)(?:_(?P<b>\d+))?$")

def parse_a_sub(p: Path):
    """
    返回 (A, sub)：
      - A.png      -> sub=3 (在最后一个时间戳)
      - A_B.png    -> sub=B (映射到时间戳)
    """
    m = PAT.match(p.stem)
    if not m:
        return (10**18, 10**18)  # 无法解析的排最后
    a = int(m.group("a"))
    if m.group("b") is None:
        sub = 3  # `A.png` 走最后一个时间点
    else:
        b = int(m.group("b"))
        sub = b
    return (a, sub)

def parse_idx(p: Path) -> int:
    a, sub = parse_a_sub(p)
    return a * 4 + sub

def collect_images(folder: Path):
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder.resolve()}")

    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if not files:
        raise FileNotFoundError(f"No images found in: {folder.resolve()}")

    # 排序：先按 (A,sub)，同键再按文件名稳定排序
    return sorted(files, key=lambda p: (parse_a_sub(p), p.name.lower()))

def main():
    files = collect_images(FAN_DIR)

    SORTED_DIR.mkdir(parents=True, exist_ok=True)

    vw = None
    video_size = None
    fourcc = cv2.VideoWriter_fourcc(*VIDEO_FOURCC)

    try:
        with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["order", "A", "sub", "idx_4000fps", "t_seconds", "filename", "path", "fps"])
            for order, p in enumerate(files):
                a, sub = parse_a_sub(p)
                idx = a * 4 + sub
                t = idx / FPS
                w.writerow([order, a, sub, idx, f"{t:.12f}", p.name, str(p.resolve()), FPS])

                if SAVE_VIDEO:
                    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                    if img is None:
                        raise RuntimeError(f"Failed to read image: {p}")

                    if vw is None:
                        H, W = img.shape[:2]
                        video_size = (W, H)
                        vw = cv2.VideoWriter(str(VIDEO_PATH), fourcc, float(VIDEO_FPS), video_size)
                        if not vw.isOpened():
                            raise RuntimeError(f"Failed to open video writer: {VIDEO_PATH}")

                    frame = img
                    if (img.shape[1], img.shape[0]) != video_size:
                        frame = cv2.resize(img, video_size, interpolation=cv2.INTER_AREA)
                    vw.write(frame)
    finally:
        if vw is not None:
            vw.release()

    print(f"[OK] Found {len(files)} images in {FAN_DIR.resolve()}")
    print(f"[OK] fps={FPS}, dt={1/FPS:.9f}s")
    print(f"[OK] CSV saved to: {OUT_CSV.resolve()}")
    if SAVE_VIDEO:
        print(f"[OK] Video saved to: {VIDEO_PATH.resolve()}")

    print("First 12 after sorting (order | A | sub | idx | t | name):")
    for i, p in enumerate(files[:12]):
        a, sub = parse_a_sub(p)
        idx = a * 4 + sub
        print(f"  {i:04d} | {a:4d} | {sub:2d} | {idx:6d} | {idx/FPS:.6f}s | {p.name}")

    if EXPORT_SORTED:
        for p in files:
            a, sub = parse_a_sub(p)
            idx = a * 4 + sub
            dst = SORTED_DIR / f"frame_{idx:03d}{p.suffix.lower()}"
            shutil.copy2(p, dst)
        print(f"[OK] Exported sorted sequence to: {SORTED_DIR.resolve()}")

if __name__ == "__main__":
    main()
