import re
import csv
import shutil
from pathlib import Path

import cv2

FPS = 4000.0
EXPORT_SORTED = True  
SAVE_VIDEO = True   
VIDEO_FPS = 30.0
VIDEO_FOURCC = "mp4v"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

PAT = re.compile(r"^(?P<a>\d+)(?:_(?P<b>\d+))?$")

def parse_a_sub(p: Path):
    m = PAT.match(p.stem)
    if not m:
        return (10**18, 10**18)
    a = int(m.group("a"))
    if m.group("b") is None:
        sub = 3 
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

    return sorted(files, key=lambda p: (parse_a_sub(p), p.name.lower()))

def process_single_folder(input_dir: Path, output_dir: Path):
    """处理单个文件夹的核心逻辑"""
    files = collect_images(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_csv = output_dir / "sequence_timestamps.csv"
    video_path = output_dir / "sequence.mp4"

    vw = None
    video_size = None
    fourcc = cv2.VideoWriter_fourcc(*VIDEO_FOURCC)

    try:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
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
                        continue

                    if vw is None:
                        H, W = img.shape[:2]
                        video_size = (W, H)
                        vw = cv2.VideoWriter(str(video_path), fourcc, float(VIDEO_FPS), video_size)

                    frame = img
                    if (img.shape[1], img.shape[0]) != video_size:
                        frame = cv2.resize(img, video_size, interpolation=cv2.INTER_AREA)
                    vw.write(frame)
    finally:
        if vw is not None:
            vw.release()

    if EXPORT_SORTED:
        for p in files:
            a, sub = parse_a_sub(p)
            idx = a * 4 + sub
            dst = output_dir / f"frame_{idx:03d}{p.suffix.lower()}"
            shutil.copy2(p, dst)
            
    print(f"[OK] Processed {input_dir.name}: {len(files)} images -> {output_dir.resolve()}")


def main():
    # ==== 基础路径设置 ====
    BASE_DIR = Path("/home/zhaoyue/Fastchange/Datasets/Scene")
    SORTED_BASE_DIR = Path("/home/zhaoyue/Fastchange/Datasets/Scene_sorted")
    
    # 遍历 1 到 10 文件夹
    for i in range(1, 11):
        input_folder = BASE_DIR / str(i)
        output_folder = SORTED_BASE_DIR / str(i)
        
        if input_folder.exists():
            print(f"\n--- Timestamping Folder: {i} ---")
            process_single_folder(input_folder, output_folder)
        else:
            print(f"Warning: Input folder not found, skipping: {input_folder}")

if __name__ == "__main__":
    main()