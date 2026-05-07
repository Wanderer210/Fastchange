import os
from typing import Optional

import numpy as np
import cv2
import h5py

def load_events_from_h5(h5_path):
    """
    从HDF5文件加载事件数据
    
    Args:
        h5_path: HDF5文件路径
    
    Returns:
        events: numpy数组，形状为[N, 4]，每行包含[timestamp, x, y, polarity]
    """
    import h5py
    with h5py.File(h5_path, 'r') as f:
        events = f['events'][:]
    print(f"加载了 {events.shape[0]} 个事件")
    print(f"时间范围: {events[0, 0]} - {events[-1, 0]} 微秒")
    print(f"空间范围: x=[{events[:, 1].min()}, {events[:, 1].max()}], y=[{events[:, 2].min()}, {events[:, 2].max()}]")
    return events

def estimate_image_dimensions(events):
    """
    根据事件数据估计图像尺寸
    """
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
    """
    可视化事件流数据

    Args:
        h5_path: 事件数据HDF5文件路径
        num_bins: 体素网格的时间分箱数量
        output_dir: 输出可视化结果的目录
        save_video: 是否保存视频（默认True）
        save_images: 是否保存PNG帧序列（默认True）
        video_fps: 输出视频帧率（默认10）
        video_fourcc: OpenCV视频编码器fourcc（默认mp4v）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载事件数据
    events = load_events_from_h5(h5_path)
    
    # 估计图像尺寸
    if width is None or height is None:
        width, height = estimate_image_dimensions(events)
    width = int(width)
    height = int(height)
    print(f"估计的图像尺寸: {width} x {height}")

    # 转换为体素网格
    print(f"转换为体素网格，时间分箱数: {num_bins}")
    voxel_grid = events_to_voxel_grid(events, num_bins, width, height)
    print(f"体素网格形状: {voxel_grid.shape}")

    filename_key = os.path.splitext(os.path.basename(h5_path))[0]

    if save_h5:
        if h5_out is None:
            h5_out = os.path.join(output_dir, f"{filename_key}_voxel.h5")
        out_shape = save_voxel_grid_h5(voxel_grid, h5_out, key="data")
        print(f"体素网格已保存到: {os.path.abspath(h5_out)}")
        print(f"保存数据集: /data shape={out_shape}")

    # 可视化并保存
    print(f"保存可视化结果到: {output_dir}")
    visual_voxel_grid_color(
        voxel_grid,
        output_dir,
        filename_key,
        save_video=save_video,
        save_images=save_images,
        video_fps=video_fps,
        video_fourcc=video_fourcc,
    )

    print(f"可视化完成！结果保存在 {output_dir} 目录中")

# 基于双线性时间插值的体素网格编码方法(源自E2VID框架)
def events_to_voxel_grid(events, num_bins, width, height):
    """
    https://github.com/uzh-rpg/rpg_e2vid/blob/master/utils/inference_utils.py
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    :param events: a [N x 4] NumPy array (np.int32) containing one event per row in the form:
        [timestamp(us), x, y, polarity(0 or 1)]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
        """

    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    # normalize the event timestamps so that they lie between 0 and num_bins
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
    pols[pols == 0] = -1  # polarity should be +1 / -1

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

#     """固定事件数量的体素网格"""
def events_to_voxel_grid_fixed_events(events, num_bins, width, height):
    voxel_grid = np.zeros((num_bins, height, width), np.float32)
    
    events_per_bin = len(events) // num_bins
    
    for bin_idx in range(num_bins):
        start_idx = bin_idx * events_per_bin
        end_idx = min((bin_idx + 1) * events_per_bin, len(events))
        
        bin_events = events[start_idx:end_idx]
        for event in bin_events:
            t, x, y, pol = event
            if pol == 0:
                pol = -1  # 0→-1, 1→+1
            voxel_grid[bin_idx, y, x] += pol
    
    return voxel_grid


def visual_voxel_grid(voxel_grid, output_folder, filename_key):
    global_min = float(np.min(voxel_grid))
    global_max = float(np.max(voxel_grid))
    denom = global_max - global_min
    if denom <= 0:
        denom = 1.0

    for i in range(voxel_grid.shape[0]):
        path = os.path.join(output_folder, '%s_%02d.png' % (filename_key, i))
        normalize_im = (voxel_grid[i] - global_min) / denom
        cv2.imwrite(path, (normalize_im * 255).astype(np.uint8))

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
    if global_positive_max <= 0:
        global_positive_max = 1.0
    if global_negative_max <= 0:
        global_negative_max = 1.0

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
                    if not vw.isOpened():
                        raise RuntimeError(f"Failed to open video writer: {video_path}")

                frame = rgb_image
                if (rgb_image.shape[1], rgb_image.shape[0]) != video_size:
                    frame = cv2.resize(rgb_image, video_size, interpolation=cv2.INTER_AREA)
                vw.write(frame)
    finally:
        if vw is not None:
            vw.release()

    if save_video:
        print(f"视频已保存: {video_path}")

def main():
    """
    主函数：处理单个指定的txt文件
    """
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='可视化事件流数据')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='输入事件数据H5文件路径（需要包含 /events 数据集）')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='输出目录路径（可选，默认为输入文件名+_visualization）')
    parser.add_argument('--bins', '-b', type=int, default=6,
                       help='时间分箱数量（可选，默认为6）')
    parser.add_argument('--fps', type=float, default=10.0,
                       help='输出视频帧率（可选，默认为10）')
    parser.add_argument('--width', type=int, default=None,
                       help='强制输出宽度（像素），不填则从events推断')
    parser.add_argument('--height', type=int, default=None,
                       help='强制输出高度（像素），不填则从events推断')
    parser.add_argument('--save-h5', action='store_true',
                       help='保存体素网格到H5，数据集为 /data，形状为 (T,2,H,W)')
    parser.add_argument('--h5-out', type=str, default=None,
                       help='输出体素网格H5路径（默认在输出目录下生成 *_voxel.h5）')
    parser.add_argument('--no-video', action='store_true',
                       help='不保存视频，仅保存PNG帧序列')
    parser.add_argument('--no-images', action='store_true',
                       help='不保存PNG帧序列，仅保存视频')

    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"错误：输入文件 {args.input} 不存在")
        return
    
    if not args.input.endswith('.h5'):
        print(f"警告：输入文件 {args.input} 不是h5文件")
    
    # 设置输出目录
    if args.output is None:
        # 使用输入文件名创建默认输出目录
        input_filename = os.path.splitext(os.path.basename(args.input))[0]
        output_dir = f"{input_filename}_visualization"
    else:
        output_dir = args.output
    
    save_video = not args.no_video
    save_images = not args.no_images

    if (not save_video) and (not save_images):
        print("错误：--no-video 与 --no-images 不能同时使用（会导致没有任何输出）")
        return

    print(f"输入文件: {args.input}")
    print(f"输出目录: {output_dir}")
    print(f"时间分箱数: {args.bins}")
    print(f"保存视频: {save_video}")
    print(f"保存PNG:  {save_images}")
    if save_video:
        print(f"视频帧率: {args.fps}")

    try:
        visualize_events(
            args.input,
            num_bins=args.bins,
            output_dir=output_dir,
            save_video=save_video,
            save_images=save_images,
            video_fps=args.fps,
            width=args.width,
            height=args.height,
            save_h5=args.save_h5,
            h5_out=args.h5_out,
        )
        print(f"\n处理完成！可视化结果保存在 {output_dir} 目录中")
    except Exception as e:
        print(f"处理文件时出错: {e}")

if __name__ == "__main__":
    main()

