import os
from typing import Optional
import argparse

import numpy as np
import cv2
import h5py

def load_events_from_h5(h5_path):
    """从HDF5文件加载事件数据"""
    with h5py.File(h5_path, 'r') as f:
        events = f['events'][:]
    print(f"  └─ 加载了 {events.shape[0]} 个事件")
    if events.shape[0] > 0:
        print(f"  └─ 时间范围: {events[0, 0]} - {events[-1, 0]} 微秒")
    return events

def estimate_image_dimensions(events):
    """根据事件数据估计图像尺寸"""
    if len(events) == 0:
        return 0, 0
    width = events[:, 1].max() + 1
    height = events[:, 2].max() + 1
    return width, height

def voxel_grid_to_2c(voxel_grid: np.ndarray) -> np.ndarray:
    pos = np.maximum(voxel_grid, 0).astype(np.float32, copy=False)
    neg = np.maximum(-voxel_grid, 0).astype(np.float32, copy=False)
    return np.stack([pos, neg], axis=1)

def save_voxel_grid_h5(voxel_grid: np.ndarray, out_h5_path: str, key: str = "data"):
    out_h5_path = str(out_h5_path)
    os.makedirs(os.path.dirname(out_h5_path) or ".", exist_ok=True)

    data_2c = voxel_grid_to_2c(voxel_grid)

    with h5py.File(out_h5_path, "w") as f:
        f.create_dataset(key, data=data_2c, compression="gzip", compression_opts=4)
        f.attrs["format"] = "voxel_grid"
        f.attrs["layout"] = "T,C,H,W"
        f.attrs["polarity_channels"] = "pos,neg"

    return data_2c.shape

def events_to_voxel_grid(events, num_bins, width, height):
    """基于双线性时间插值的体素网格编码方法"""
    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    last_stamp = events[-1, 0]
    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    ts = events[:, 0].astype(np.float32)
    ts = (num_bins - 1) * (ts - first_stamp) / deltaT
    xs = events[:, 1].astype(np.int32)
    ys = events[:, 2].astype(np.int32)
    pols = events[:, 3].astype(np.float32)
    pols[pols == 0] = -1 

    tis = ts.astype(np.int32)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + tis[valid_indices] * width * height, vals_left[valid_indices])

    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))
    return voxel_grid

def visual_voxel_grid_color(
    voxel_grid,
    output_folder,
    filename_key,
    save_video=True,
    save_images=True,
    video_fps=10.0,
    video_fourcc="mp4v",
):
    vw = None
    video_size = None
    video_path = os.path.join(output_folder, f"{filename_key}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*video_fourcc)

    global_positive_max = float(np.max(np.maximum(voxel_grid, 0)))
    global_negative_max = float(np.max(np.maximum(-voxel_grid, 0)))
    if global_positive_max <= 0: global_positive_max = 1.0
    if global_negative_max <= 0: global_negative_max = 1.0

    try:
        for i in range(voxel_grid.shape[0]):
            path = os.path.join(output_folder, "%s_%02d.png" % (filename_key, i))
            current_voxel = voxel_grid[i]
            height, width = current_voxel.shape
            rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

            positive_events = np.maximum(current_voxel, 0)
            negative_events = np.maximum(-current_voxel, 0)

            if np.max(positive_events) > 0:
                positive_normalized = (positive_events / global_positive_max * 255).astype(np.uint8)
                rgb_image[:, :, 2] = positive_normalized

            if np.max(negative_events) > 0:
                negative_normalized = (negative_events / global_negative_max * 255).astype(np.uint8)
                rgb_image[:, :, 0] = negative_normalized

            if save_images:
                cv2.imwrite(path, rgb_image)

            if save_video:
                if vw is None:
                    video_size = (width, height)
                    vw = cv2.VideoWriter(str(video_path), fourcc, float(video_fps), video_size)
                frame = rgb_image
                if (rgb_image.shape[1], rgb_image.shape[0]) != video_size:
                    frame = cv2.resize(rgb_image, video_size, interpolation=cv2.INTER_AREA)
                vw.write(frame)
    finally:
        if vw is not None:
            vw.release()

    if save_video:
        print(f"  └─ 视频已保存: {video_path}")


def visualize_events(
    h5_path,
    num_bins=6,
    output_dir="visualizations",
    save_video=True,
    save_images=True,
    video_fps=10.0,
    video_fourcc="mp4v",
    width: Optional[int] = None,
    height: Optional[int] = None,
    save_h5: bool = False,
    h5_out: Optional[str] = None,
):
    """核心可视化与网格化函数"""
    os.makedirs(output_dir, exist_ok=True)
    events = load_events_from_h5(h5_path)
    
    if len(events) == 0:
        print(f"  └─ 警告: {h5_path} 中没有事件数据，跳过")
        return False

    if width is None or height is None:
        w_est, h_est = estimate_image_dimensions(events)
        width = width or w_est
        height = height or h_est
    
    width, height = int(width), int(height)
    print(f"  └─ 转换为体素网格，图像尺寸: {width}x{height}，分箱数: {num_bins}")
    
    voxel_grid = events_to_voxel_grid(events, num_bins, width, height)
    filename_key = os.path.splitext(os.path.basename(h5_path))[0]

    if save_h5:
        if h5_out is None:
            h5_out = os.path.join(output_dir, f"{filename_key}_voxel.h5")
        out_shape = save_voxel_grid_h5(voxel_grid, h5_out, key="data")
        print(f"  └─ 体素网格H5已保存: {os.path.abspath(h5_out)} | Shape: {out_shape}")

    if save_video or save_images:
        visual_voxel_grid_color(
            voxel_grid,
            output_dir,
            filename_key,
            save_video=save_video,
            save_images=save_images,
            video_fps=video_fps,
            video_fourcc=video_fourcc,
        )
    return True


def process_flat_folders(dataset_dir, out_subfolder_name, args):
    """遍历扁平文件夹结构（如 1 到 10 文件夹），逐个生成体素网格"""
    if not os.path.exists(dataset_dir):
        print(f"目录不存在: {dataset_dir}")
        return
        
    folder_names = os.listdir(dataset_dir)
    def sort_key(x):
        try: return int(x)
        except ValueError: return x
        
    for folder_name in sorted(folder_names, key=sort_key):
        folder_path = os.path.join(dataset_dir, folder_name)
        if not os.path.isdir(folder_path): continue
            
        input_h5 = os.path.join(folder_path, "events.h5")
        if not os.path.exists(input_h5):
            print(f"警告: 找不到事件文件 {input_h5}，跳过")
            continue
            
        print(f"\n============ 处理: {dataset_dir}/{folder_name} ============")
        output_dir = os.path.join(folder_path, out_subfolder_name)
        
        try:
            visualize_events(
                input_h5,
                num_bins=args.bins,
                output_dir=output_dir,
                save_video=not args.no_video,
                save_images=not args.no_images,
                video_fps=args.fps,
                width=args.width,
                height=args.height,
                save_h5=args.save_h5,
                h5_out=args.h5_out if hasattr(args, 'h5_out') else None,
            )
        except Exception as e:
            print(f"处理文件时出错: {e}")


def main():
    parser = argparse.ArgumentParser(description='将事件流数据转换为体素网格(Voxel Grid)并可视化')
    subparsers = parser.add_subparsers(dest='mode', help='运行模式')
    
    # ======== 模式 1: 单一文件处理 ========
    parser_single = subparsers.add_parser('single', help='处理单个 events.h5 文件')
    parser_single.add_argument('--input', '-i', type=str, required=True, help='输入事件数据H5文件路径')
    parser_single.add_argument('--output', '-o', type=str, default=None, help='输出目录路径')
    
    # ======== 模式 2: 扁平目录处理 ========
    parser_flat = subparsers.add_parser('flat_sequence', help='遍历扁平目录结构 (如 1~10 文件夹)')
    parser_flat.add_argument('--dataset_dir', required=True, help='事件流数据集根目录')
    parser_flat.add_argument('--out_subfolder', type=str, default='voxel_results', help='生成的结果放在文件夹内的子目录名称')
    
    # ======== 模式 3: 双向批量处理 ========
    parser_dual = subparsers.add_parser('dual_process', help='一键批量生成正常(Normal)与异常(Light)体素网格')
    parser_dual.add_argument('--normal_dir', type=str, default='/home/zhaoyue/Fastchange/Datasets/Scene-normal-event', help='正常事件流目录')
    parser_dual.add_argument('--light_dir', type=str, default='/home/zhaoyue/Fastchange/Datasets/Scene-light-event', help='异常事件流目录')
    parser_dual.add_argument('--out_subfolder', type=str, default='voxel_results_768', help='每个编号文件夹内保存结果的子目录名称')

    # 所有模式的通用参数
    for p in [parser_single, parser_flat, parser_dual]:
        p.add_argument('--bins', '-b', type=int, default=10, help='时间分箱数量（默认10）')
        p.add_argument('--fps', type=float, default=10.0, help='输出视频帧率（默认10）')
        p.add_argument('--width', type=int, default=768, help='强制输出宽度（像素，默认768）')
        p.add_argument('--height', type=int, default=768, help='强制输出高度（像素，默认768）')
        p.add_argument('--save-h5', action='store_true', default=True, help='默认开启：保存体素网格到H5')
        p.add_argument('--no-video', action='store_true', help='不保存视频')
        p.add_argument('--no-images', action='store_true', help='不保存PNG帧序列')

    # 单独为 single 添加 h5-out 选项
    parser_single.add_argument('--h5-out', type=str, default=None, help='输出体素网格H5路径')

    args = parser.parse_args()
    
    if args.mode is None:
        parser.print_help()
        return

    if (args.no_video) and (args.no_images) and (not args.save_h5):
        print("错误：你禁用了视频、图片和H5输出，没有任何数据被保存！")
        return

    print("===== Voxel Grid Generator =====")
    print(f"Bins: {args.bins}, Width: {args.width}, Height: {args.height}")
    
    if args.mode == 'single':
        output_dir = args.output if args.output else f"{os.path.splitext(os.path.basename(args.input))[0]}_visualization"
        visualize_events(
            args.input,
            num_bins=args.bins,
            output_dir=output_dir,
            save_video=not args.no_video,
            save_images=not args.no_images,
            video_fps=args.fps,
            width=args.width,
            height=args.height,
            save_h5=args.save_h5,
            h5_out=args.h5_out,
        )

    elif args.mode == 'flat_sequence':
        process_flat_folders(args.dataset_dir, args.out_subfolder, args)
        
    elif args.mode == 'dual_process':
        print("\n====== [阶段 1/2] 开始处理【正常事件数据】 ======")
        process_flat_folders(args.normal_dir, args.out_subfolder, args)
        
        print("\n====== [阶段 2/2] 开始处理【异常光照事件数据】 ======")
        process_flat_folders(args.light_dir, args.out_subfolder, args)

    print("\n✅ 所有处理完成！")

if __name__ == "__main__":
    main()