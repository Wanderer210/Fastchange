import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import h5py
from pathlib import Path
from typing import Optional, Union
from spikingjelly.activation_based import neuron, surrogate, functional, layer

# ==========================================
# 1. 定义自适应阈值神经元 (ALIF Node) - 已修正
# ==========================================
class ALIFNode(neuron.BaseNode):
    """
    支持 [N, T, ...] 输入格式的自适应阈值神经元
    修正点：将 'v' 重命名为 'v_mem' 以避免与 BaseNode 冲突
    """
    def __init__(self, v_threshold=1.0, tau=2.0, beta=1.0, detach_reset=True):
        super().__init__(v_threshold=v_threshold, v_reset=0., surrogate_function=surrogate.ATan(), detach_reset=detach_reset, step_mode='m')
        
        self.tau = tau
        self.decay = 1.0 - (1.0 / tau)
        self.beta = beta
        self.trace_decay = 0.9  
        
        # --- 修正 1: 注册 'v_mem' 而不是 'v' ---
        self.register_memory('v_mem', 0.)
        self.register_memory('trace', 0.)

    def forward(self, x: torch.Tensor):
        if self.step_mode == 's':
            return self.single_step_forward(x)
        
        time_dim = 1 
        T = x.shape[time_dim]
        
        spikes = []
        for t in range(T):
            xt = x.select(time_dim, t) 
            s = self.single_step_forward(xt)
            spikes.append(s)
        
        return torch.stack(spikes, dim=time_dim)

    def single_step_forward(self, x: torch.Tensor):
        # --- 修正 2: 使用 self.v_mem 进行计算 ---
        
        # 1. 积分 (Charge)
        self.v_mem = self.v_mem * self.decay + x
        
        # 2. 计算自适应阈值
        adaptive_thresh = self.v_threshold + self.beta * self.trace
        
        if not isinstance(adaptive_thresh, torch.Tensor):
            adaptive_thresh = torch.as_tensor(adaptive_thresh, device=x.device, dtype=x.dtype)
        
        # 3. 发放 (Fire)
        spike = self.surrogate_function(self.v_mem - adaptive_thresh)
        
        # 4. 重置 (Reset)
        self.v_mem = self.v_mem * (1. - spike) 
        
        # 5. 更新 Trace
        self.trace = self.trace * self.trace_decay + spike
        
        return spike

# ==========================================
# 2. 网络构建 (Net)
# ==========================================
class Net(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.flatten = layer.Flatten()
        
        flat_dim = input_shape[0] * input_shape[1] * input_shape[2]
        print(f"Network Initialized. Input Shape: {input_shape}, Flatten Dim: {flat_dim}")
        
        self.fc = layer.Linear(flat_dim, 128)
        self.alif = ALIFNode(v_threshold=1.0, beta=2.0)

    def forward(self, x):
        T = x.shape[0]
        
        monitor = {
            'spike': [],
            'v_mem': [],
            'thresh': []
        }
        
        for t in range(T):
            xt = x[t] 
            out = self.flatten(xt)
            out = self.fc(out)
            
            s = th = None 
            
            # 由于我们手动在 loop 里调用 ALIF，我们需要使用 single_step_forward 
            # 或者临时把 mode 改为 's'。
            # 为了简单起见，且因为 ALIFNode 实现了 step_mode='m' 逻辑，
            # 在手动循环 T 时，我们应该只让它跑一步。
            
            # 方法 A: 临时切换 step_mode (SpikingJelly 标准做法)
            self.alif.step_mode = 's'
            s = self.alif(out)
            
            # 获取阈值用于记录 (ALIF 内部没有直接返回 thresh，我们需要重新计算一下用于显示，或者修改 forward 返回)
            # 为了不破坏上面的类结构，我们在这里手动计算一下当前的阈值用于可视化
            current_thresh = self.alif.v_threshold + self.alif.beta * self.alif.trace
            
            monitor['spike'].append(s.detach().cpu())
            
            # --- 修正 3: 读取 self.alif.v_mem ---
            monitor['v_mem'].append(self.alif.v_mem.detach().cpu())
            
            # 确保阈值是 tensor
            if not isinstance(current_thresh, torch.Tensor):
                 current_thresh = torch.as_tensor(current_thresh)
            monitor['thresh'].append(current_thresh.detach().cpu())
            
        for k in monitor:
            monitor[k] = torch.stack(monitor[k])
            
        return monitor

# ==========================================
# 3. 数据加载 (保持不变)
# ==========================================
def load_h5_voxel_grid(file_path, key='data'):
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with h5py.File(file_path, 'r') as f:
        if key not in f.keys():
            print(f"Warning: Key '{key}' not found. Available: {list(f.keys())}")
            key = list(f.keys())[0]
            
        data = f[key][:] 

    tensor_data = torch.from_numpy(data).float()

    if tensor_data.ndim == 4: 
        tensor_data = tensor_data.unsqueeze(1)
    elif tensor_data.ndim == 3:
        tensor_data = tensor_data.unsqueeze(1).unsqueeze(1)
    
    print(f"Loaded {file_path.name}: Shape {tensor_data.shape}, Max Val: {tensor_data.max()}")
    return tensor_data

# ==========================================
# 4. 运行实验
# ==========================================
def run_comparison(normal_path, flash_path, save_path: Optional[Union[str, Path]] = None):
    # 1. 加载数据
    print(f"Loading Normal Data from: {normal_path}")
    data_normal = load_h5_voxel_grid(normal_path)
    
    print(f"Loading Flash Data from: {flash_path}")
    data_flash = load_h5_voxel_grid(flash_path)

    # 校验
    assert data_normal.shape == data_flash.shape, "Error: Shapes must match!"
    T = data_normal.shape[0]
    C, H, W = data_normal.shape[2], data_normal.shape[3], data_normal.shape[4]

    denom = float(C * H * W)

    normal_event_density = (data_normal.abs().sum(dim=(1, 2, 3, 4)) / denom).cpu().numpy()
    flash_event_density = (data_flash.abs().sum(dim=(1, 2, 3, 4)) / denom).cpu().numpy()

    normal_nonzero_ratio = (data_normal != 0).float().mean(dim=(1, 2, 3, 4)).cpu().numpy()
    flash_nonzero_ratio = (data_flash != 0).float().mean(dim=(1, 2, 3, 4)).cpu().numpy()

    print("Per-timestep event density: abs-sum / (C*H*W)")
    for t in range(T):
        print(
            f"t={t:03d}  normal={normal_event_density[t]:.6g}  flash={flash_event_density[t]:.6g}"
            f"  nonzero(normal)={normal_nonzero_ratio[t]:.6g}  nonzero(flash)={flash_nonzero_ratio[t]:.6g}"
        )

    time_axis = np.arange(T)

    fig_stats, axs_stats = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    normal_density_plot = np.clip(normal_event_density, 1e-12, None)
    flash_density_plot = np.clip(flash_event_density, 1e-12, None)

    axs_stats[0].plot(time_axis, normal_density_plot, 'g-o', label='Normal')
    axs_stats[0].plot(time_axis, flash_density_plot, 'r-o', label='Flash')
    axs_stats[0].set_yscale('log')
    axs_stats[0].set_ylabel('abs-sum / (C*H*W)')
    axs_stats[0].set_title('Per-timestep Event Density (log scale)')
    axs_stats[0].grid(True, alpha=0.3)
    axs_stats[0].legend()

    axs_stats[1].plot(time_axis, normal_nonzero_ratio, 'g-o', label='Nonzero ratio (Normal)')
    axs_stats[1].plot(time_axis, flash_nonzero_ratio, 'r-o', label='Nonzero ratio (Flash)')
    axs_stats[1].set_ylim(0.0, 1.05)
    axs_stats[1].set_ylabel('Nonzero ratio')
    axs_stats[1].set_xlabel('Time bin')
    axs_stats[1].grid(True, alpha=0.3)
    axs_stats[1].legend()

    fig_stats.tight_layout()

    if save_path is None:
        stats_save_path = Path(__file__).with_name('snn_h5_stats.png')
    else:
        _p = Path(save_path)
        if _p.is_dir():
            stats_save_path = _p / 'snn_h5_stats.png'
        else:
            suffix = _p.suffix if _p.suffix else '.png'
            stats_save_path = _p.with_name(f"{_p.stem}_stats{suffix}")

    fig_stats.savefig(stats_save_path, dpi=200, bbox_inches='tight')
    plt.close(fig_stats)
    print(f"Saved stats figure to: {stats_save_path.resolve()}")

    # 2. 初始化
    net = Net(input_shape=(C, H, W))
    
    # --- Normal ---
    print("Processing Normal Data...")
    functional.reset_net(net) 
    rec_normal = net(data_normal)
    
    # --- Flash ---
    print("Processing Flash Data...")
    functional.reset_net(net) 
    rec_flash = net(data_flash)
    
    # ================= 绘图 =================
    fig, axs = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
    time_axis = np.arange(T) 
    
    cols = ['Normal Stream', 'Flash Stream']
    for ax, col in zip(axs[0], cols):
        ax.set_title(col, fontsize=14, fontweight='bold')

    # 1. Input
    axs[0, 0].plot(time_axis, data_normal.sum(dim=(1,2,3,4)).numpy(), 'g-o')
    axs[0, 1].plot(time_axis, data_flash.sum(dim=(1,2,3,4)).numpy(), 'r-o')
    axs[0, 0].set_ylabel('Voxel Sum')

    # 2. Raster
    def plot_raster(ax, spike_tensor, color):
        spikes = spike_tensor.squeeze(1).numpy() 
        rows, cols = np.where(spikes > 0)
        ax.scatter(rows, cols, s=5, c=color)
        ax.set_ylim(0, 128)
        ax.set_ylabel('Neuron Index')
        ax.grid(True, alpha=0.3, linestyle='--')

    plot_raster(axs[1, 0], rec_normal['spike'], 'blue')
    plot_raster(axs[1, 1], rec_flash['spike'], 'orange')
    
    # 3. V_mem
    v_mean_norm = rec_normal['v_mem'].mean(dim=(1,2)).numpy()
    v_mean_flash = rec_flash['v_mem'].mean(dim=(1,2)).numpy()
    axs[2, 0].plot(time_axis, v_mean_norm, 'b-o')
    axs[2, 1].plot(time_axis, v_mean_flash, 'orange', marker='o')
    axs[2, 0].set_ylabel('Mean V_mem')
    axs[2, 0].grid(True)
    axs[2, 1].grid(True)

    # 4. Threshold
    th_mean_norm = rec_normal['thresh'].mean(dim=(1,2)).numpy()
    th_mean_flash = rec_flash['thresh'].mean(dim=(1,2)).numpy()
    
    axs[3, 0].plot(time_axis, th_mean_norm, 'k--', label='Threshold')
    axs[3, 0].plot(time_axis, v_mean_norm, 'b-', alpha=0.3)
    
    axs[3, 1].plot(time_axis, th_mean_flash, 'k--', linewidth=2, label='Adaptive Threshold')
    axs[3, 1].plot(time_axis, v_mean_flash, 'orange', alpha=0.3)
    axs[3, 1].legend()
    axs[3, 1].set_ylabel('Threshold Level')
    
    from matplotlib.ticker import MaxNLocator
    axs[3, 0].xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()

    if save_path is None:
        save_path = Path(__file__).with_name('snn_h5_comparison.png')
    
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Saved figure to: {save_path.resolve()}")

if __name__ == "__main__":
    NORMAL_H5_PATH = "/home/zy/zhaoyue/Fastchange/Datasets/Fan_sorted_events/my_voxel_results_768/events_voxel.h5" 
    FLASH_H5_PATH = "/home/zy/zhaoyue/Fastchange/Datasets/Fan_events/my_voxel_results_768/events_voxel.h5" 
    
    # 检测文件是否存在，不存在则跳过 (Demo用)
    if not Path(NORMAL_H5_PATH).exists():
        print("File not found, skipping execution.")
    else:
        run_comparison(NORMAL_H5_PATH, FLASH_H5_PATH)